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
        self.vector_verify = VectorVerification(config.vector, hardware_profile=self.hardware)
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

        # If require_person is enabled and no persons are present in the frame, skip firearm detection to save CPU computation
        if self.config.detector.require_person and len(tracked_persons) == 0:
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
        # STEP 4: CROP + RESIZE + DEEP GUN CLASSIFIER (Gun vs Not-Gun)
        # Verifies candidate gun crops against false alarms (bottles, phones, tools)
        # Adds contextual padding so the classifier captures hand grip relationship
        # =========================================================================
        verified_guns = []
        crop_pad_ratio = getattr(self.config.vector, "crop_padding_ratio", 0.20)

        for gun in candidate_guns:
            if not is_sim and frame is not None and hasattr(frame, "shape"):
                gx1, gy1, gx2, gy2 = gun["bbox"]
                h_img, w_img = frame.shape[:2]
                box_w = gx2 - gx1
                box_h = gy2 - gy1
                
                # Contextual margin/padding
                pad_x = int(box_w * crop_pad_ratio)
                pad_y = int(box_h * crop_pad_ratio)
                cx1 = max(0, gx1 - pad_x)
                cy1 = max(0, gy1 - pad_y)
                cx2 = min(w_img, gx2 + pad_x)
                cy2 = min(h_img, gy2 + pad_y)

                gun_crop = frame[cy1:cy2, cx1:cx2] if (cx2 > cx1 and cy2 > cy1) else None
                vec_score, matched_cat = self.vector_verify.verify_crop(gun_crop)
            else:
                crop_sim = frame.get("crop_sim_info", {}) if is_sim else {}
                vec_score, matched_cat = self.vector_verify.verify_crop(crop_sim)

            gun["vector_score"] = vec_score
            gun["vector_match"] = matched_cat

            # Filter out non-firearm objects:
            # Must have vector_score >= similarity_threshold AND matched_cat in firearm categories
            is_negative = any(neg in matched_cat.lower() for neg in self.vector_verify.hard_negatives)
            is_firearm_cat = matched_cat.lower() in [c.lower() for c in self.vector_verify.firearm_categories]
            is_verified = (not is_negative) and is_firearm_cat and (vec_score >= self.config.vector.similarity_threshold)
            
            if is_verified:
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
                is_temporally_valid = True

            if is_sim:
                is_temporally_valid = True
                if "sim_duration_sec" in frame:
                    gun_duration_sec = frame["sim_duration_sec"]

            # Pose context & wrist check
            if person_bbox is not None:
                pose_info = self.context_checker.analyze_pose(frame, person_bbox, gun["bbox"])
                pose_risk_factor = pose_info["pose_risk_factor"]
                is_in_hand = pose_info.get("is_in_hand", False)
            else:
                pose_info = {
                    "context": "unattended",
                    "pose_risk_factor": 0.5,
                    "is_in_hand": False,
                    "wrist_distance": 999.0
                }
                pose_risk_factor = 0.5
                is_in_hand = False

            # Strict Verification: If require_person is enabled, only confirm if firearm is in hand and associated with a person
            if self.config.detector.require_person and (not assoc_pid or not is_in_hand):
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

            if assoc_pid and is_in_hand:
                self.tracker.mark_suspect(assoc_pid, is_armed=True)
                active_threat_track_ids.add(assoc_pid)
                if assoc_pid not in self.tracker.reid_gallery:
                    self.tracker.reid_gallery[assoc_pid] = {
                        "vector": None,
                        "last_bbox": tracked_persons[assoc_pid].bbox if assoc_pid in tracked_persons else (0, 0, 0, 0),
                        "is_suspect": True,
                        "previously_armed": True,
                        "last_armed_timestamp": time.time()
                    }

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
        # STEP 8: PERSON STATUS & CONCEALED WEAPON STICKY SUSPECT TRACKING
        # =========================================================================
        persons_payload = []
        for pid, p in tracked_persons.items():
            is_currently_armed = pid in active_threat_track_ids
            is_concealed = p.previously_armed and not is_currently_armed

            persons_payload.append({
                "track_id": pid,
                "bbox": p.bbox,
                "is_suspect": is_currently_armed or is_concealed,
                "previously_armed": p.previously_armed,
                "is_concealed": is_concealed
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
