import logging
import uuid
import time
from typing import Dict, List, Any

from .config import SystemConfig
from .quality_gate import FrameQualityGate
from .tracking import PersonTracker
from .ensemble import DetectionEnsemble
from .vector_verify import VectorVerification
from .temporal import TemporalValidator
from .context_check import PoseContextChecker
from .association import PersonGunAssociation
from .risk_score import ThreatRiskScorer
from .alert_system import AlertSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GunDetectionPipeline")

class GunDetectionPipeline:
    """
    Step-by-step Multi-Stage Gun Detection & Verification Pipeline:
    Step 1: Frame Quality Gate (adjust dynamic thresholds)
    Step 2: Person Detection & Tracking (persistent ID tracking)
    Step 3: Gun Detection Ensemble (primary YOLO + secondary models)
    Step 4: Crop & Feature/Vector Verification (filter out false alarms: bottles, phones, etc.)
    Step 5: Multi-Frame Temporal Validation (filter out single-frame flickers)
    Step 6: Person-Gun Association (link verified gun to person)
    Step 7: Pose / Context & Risk Scoring (compute threat risk)
    Step 8: Alert Dispatch & Sticky Suspect Memory
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.hardware = getattr(config, "hardware", None)
        if self.hardware is None:
            self.hardware = SystemConfig.detect_hardware(config.detector.device)
            self.config.hardware = self.hardware

        logger.info(f"Initialized Pipeline on {self.hardware.device_name} (Mode: {'High-End GPU' if self.hardware.is_high_end_gpu else ('GPU' if self.hardware.has_cuda else 'CPU Adaptive')}, Resolution: {self.hardware.imgsz}px, FP16: {self.hardware.use_half})")

        self.quality_gate = FrameQualityGate(config.quality)
        self.tracker = PersonTracker(config.detector, hardware_profile=self.hardware)
        self.ensemble = DetectionEnsemble(config.detector, hardware_profile=self.hardware)
        self.vector_verify = VectorVerification(config.vector)
        self.temporal_validator = TemporalValidator(config.temporal)
        self.context_checker = PoseContextChecker(hardware_profile=self.hardware)
        self.associator = PersonGunAssociation()
        self.risk_scorer = ThreatRiskScorer(config.risk)
        self.alert_system = AlertSystem()

    def process_frame(self, frame, location_risk: float = 0.0) -> Dict[str, Any]:
        """
        Runs each pipeline stage strictly step-by-step in logical order.
        """
        is_sim = isinstance(frame, dict)

        # =========================================================================
        # STEP 1: FRAME QUALITY GATE
        # =========================================================================
        q_metrics = self.quality_gate.process(frame)
        threshold_offset = q_metrics["threshold_offset"]

        # =========================================================================
        # STEP 2: PERSON DETECTION + PERSISTENT TRACKING
        # =========================================================================
        person_detections = []
        if is_sim:
            person_detections = frame.get("people", [])
        
        tracked_persons = self.tracker.update(frame, person_detections)
        active_track_ids = list(tracked_persons.keys())

        # =========================================================================
        # STEP 3: GUN DETECTION ENSEMBLE (ONLY RUN IF PEOPLE DETECTED TO SAVE CPU)
        # =========================================================================
        candidate_guns = []
        watch_state = False

        # If no persons are present in the frame, skip firearm detection to save CPU computation
        if len(tracked_persons) == 0 and not is_sim:
            return {
                "persons": [],
                "detections": [],
                "quality": q_metrics
            }

        if self.config.degraded_mode or not self.config.enable_ensemble:
            primary_threshold = self.config.detector.primary_conf_threshold
            raw_detections = frame.get("detections_primary", []) if is_sim else []
            for det in raw_detections:
                if det["confidence"] >= primary_threshold:
                    candidate_guns.append({
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "class": det.get("class", "gun"),
                        "detector": "degraded_yolo_only"
                    })
        else:
            candidate_guns, watch_state = self.ensemble.detect(frame, threshold_offset)

        # =========================================================================
        # STEP 4: CROP & VECTOR / FEATURE VERIFICATION
        # Verify candidate gun crops against false alarms (bottles, phones, etc.)
        # =========================================================================
        verified_guns = []
        for gun in candidate_guns:
            if not is_sim and frame is not None and hasattr(frame, "shape"):
                gx1, gy1, gx2, gy2 = gun["bbox"]
                h_img, w_img = frame.shape[:2]
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(w_img, gx2), min(h_img, gy2)
                gun_crop = frame[gy1:gy2, gx1:gx2] if (gx2 > gx1 and gy2 > gy1) else None
                vec_score, matched_cat = self.vector_verify.verify_crop(gun_crop)
            else:
                crop_sim = frame.get("crop_sim_info", {}) if is_sim else {}
                vec_score, matched_cat = self.vector_verify.verify_crop(crop_sim)

            gun["vector_score"] = vec_score
            gun["vector_match"] = matched_cat

            # Filter out objects strongly matching hard negatives
            if matched_cat not in self.vector_verify.hard_negatives:
                verified_guns.append(gun)

        # =========================================================================
        # STEP 5: PERSON-GUN ASSOCIATION
        # Link verified guns to specific tracked people
        # =========================================================================
        associations = self.associator.associate(tracked_persons, verified_guns)
        associated_guns = {id(g): pid for pid, g in associations}
        gun_associated_track_ids = list(set(associated_guns.values()))

        # =========================================================================
        # STEP 6: MULTI-FRAME TEMPORAL VALIDATION & EXPOSURE DURATION
        # =========================================================================
        temporal_valid_map = self.temporal_validator.validate(active_track_ids, gun_associated_track_ids)

        # =========================================================================
        # STEP 7: POSE CONTEXT, THREAT RISK SCORING & ALERTS
        # =========================================================================
        detections_results = []
        active_threat_track_ids = set()

        for gun in verified_guns:
            assoc_pid = associated_guns.get(id(gun))
            gun_duration_sec = 0.0
            person_bbox = None
            is_temporally_valid = False

            if assoc_pid:
                person = tracked_persons[assoc_pid]
                person_bbox = person.bbox
                t_info = temporal_valid_map.get(assoc_pid, {})
                is_temporally_valid = t_info.get("is_valid", False)
                gun_duration_sec = t_info.get("duration_sec", 0.0)
                track_id = assoc_pid
            else:
                track_id = f"gun_unassociated_{uuid.uuid4().hex[:4]}"
                is_temporally_valid = False

            if is_sim:
                is_temporally_valid = True
                if "sim_duration_sec" in frame:
                    gun_duration_sec = frame["sim_duration_sec"]

            # Pose context & wrist check
            pose_info = self.context_checker.analyze_pose(frame, person_bbox, gun["bbox"])
            pose_risk_factor = pose_info["pose_risk_factor"]
            is_in_hand = pose_info.get("is_in_hand", False)

            # Strict Verification: Only confirm if firearm is in hand and associated with a person
            if not is_sim and (not assoc_pid or not is_in_hand):
                # Ignore unassociated objects or objects not held in hand
                continue

            # Track stability
            track_stability = 0.8 if is_temporally_valid else 0.5

            # Compute comprehensive multi-stage threat score
            metrics = {
                "detection_conf": gun["confidence"],
                "pose_risk": pose_risk_factor,
                "location_risk": location_risk if location_risk > 0.0 else 0.0,
                "track_stability": track_stability,
                "vector_score": gun.get("vector_score", 0.0)
            }
            risk_result = self.risk_scorer.compute(metrics)
            total_threat_score = risk_result["score"]

            # Multi-Stage Escalation Check:
            # A person ONLY turns RED (DANGER) if:
            # 1. Total threat score passes high threshold (>= 0.75)
            # 2. Firearm detector confidence is high (>= 0.35)
            # 3. Vector verification confirmed mechanical firearm features (vector_score >= 0.65)
            # 4. Weapon has been persistently visible for >= danger_duration_seconds (3.0s)
            is_high_threat = (total_threat_score >= self.config.risk.high_threshold and 
                              gun["confidence"] >= 0.35 and 
                              gun.get("vector_score", 0.0) >= 0.65)

            if gun_duration_sec >= self.config.temporal.danger_duration_seconds and is_high_threat:
                risk_result["level"] = "DANGER"
            elif is_high_threat or gun["confidence"] >= 0.30:
                risk_result["level"] = "HIGH"
            else:
                risk_result["level"] = "MEDIUM"

            if assoc_pid:
                self.tracker.mark_suspect(assoc_pid, is_armed=True)
                active_threat_track_ids.add(assoc_pid)

            dispatch_outcome = self.alert_system.dispatch(track_id, risk_result, frame)

            # Output verified detection immediately so gun bounding box is drawn
            detections_results.append({
                "track_id": track_id,
                "gun_bbox": gun["bbox"],
                "person_bbox": person_bbox,
                "class": gun.get("class", "gun"),
                "confidence": gun["confidence"],
                "temporal_valid": is_temporally_valid,
                "gun_duration_sec": gun_duration_sec,
                "vector_match": gun.get("vector_match", "unknown"),
                "vector_score": gun.get("vector_score", 0.0),
                "pose_context": pose_info["context"],
                "is_in_hand": is_in_hand,
                "wrist_distance": pose_info.get("wrist_distance", 0.0),
                "risk_score": risk_result["score"],
                "risk_level": risk_result["level"],
                "action": dispatch_outcome["action_taken"],
                "evidence_hash": dispatch_outcome.get("evidence_hash", None),
                "degraded_mode": self.config.degraded_mode
            })

        # =========================================================================
        # STEP 8: PERSON STATUS - ONLY ACTIVE ARMED INDIVIDUALS ARE SUSPECTS
        # (Remove false positive sticky suspect memory when person has no gun)
        # =========================================================================
        persons_payload = []
        for pid, p in tracked_persons.items():
            # A person is ONLY suspect if they are ACTIVELY carrying a verified gun in the current frame
            is_currently_armed = pid in active_threat_track_ids
            
            if not is_currently_armed:
                p.is_suspect = False
                p.previously_armed = False

            persons_payload.append({
                "track_id": pid,
                "bbox": p.bbox,
                "is_suspect": is_currently_armed,
                "previously_armed": False,
                "is_concealed": False
            })

        return {
            "persons": persons_payload,
            "detections": detections_results,
            "quality": q_metrics
        }

    def clear_suspect_status(self, track_id: str):
        """Operator audit clearance protocol method."""
        self.tracker.clear_suspect(track_id)
        logger.info(f"AUDIT PROTOCOL: Operator cleared suspect status for track ID '{track_id}'.")
