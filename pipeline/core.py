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
    def __init__(self, config: SystemConfig):
        self.config = config
        
        self.quality_gate = FrameQualityGate(config.quality)
        self.tracker = PersonTracker()
        self.ensemble = DetectionEnsemble(config.detector)
        self.vector_verify = VectorVerification(config.vector)
        self.temporal_validator = TemporalValidator(config.temporal)
        self.context_checker = PoseContextChecker()
        self.associator = PersonGunAssociation()
        self.risk_scorer = ThreatRiskScorer(config.risk)
        self.alert_system = AlertSystem()

    def process_frame(self, frame, location_risk: float = 0.5) -> Dict[str, Any]:
        """
        Runs the full gun detection and verification pipeline on a single frame.
        """
        detections_results = []

        # 1. FRAME QUALITY GATE
        q_metrics = self.quality_gate.process(frame)
        threshold_offset = q_metrics["threshold_offset"]

        # 2. PERSON DETECTION + TRACKING
        person_detections = []
        if isinstance(frame, dict):
            person_detections = frame.get("people", [])
        
        tracked_persons = self.tracker.update(frame, person_detections)
        active_track_ids = list(tracked_persons.keys())

        # DEGRADED-MODE FALLBACK CHECK
        is_degraded = self.config.degraded_mode
        if not self.config.enable_vector_db or not self.config.enable_ensemble:
            is_degraded = True

        detected_guns = []
        watch_state = False

        if is_degraded:
            primary_threshold = self.config.detector.primary_conf_threshold
            raw_detections = []
            if isinstance(frame, dict):
                raw_detections = frame.get("detections_primary", [])
            
            for det in raw_detections:
                if det["confidence"] >= primary_threshold:
                    detected_guns.append({
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "class": det.get("class", "gun"),
                        "detector": "degraded_yolo_only"
                    })
        else:
            detected_guns, watch_state = self.ensemble.detect(frame, threshold_offset)

        # 3. PERSON-GUN ASSOCIATION
        associations = self.associator.associate(tracked_persons, detected_guns)
        associated_guns = {id(gun): track_id for track_id, gun in associations}
        gun_track_ids = list(set(associated_guns.values()))

        # 4. MULTI-FRAME TEMPORAL VALIDATION & DURATION
        temporal_valid_map = self.temporal_validator.validate(active_track_ids, gun_track_ids)

        # 5. CROP & VECTOR SIMILARITY + POSE CONTEXT + RISK SCORING
        for gun in detected_guns:
            track_id = associated_guns.get(id(gun))
            gun_duration_sec = 0.0

            if track_id:
                person = tracked_persons[track_id]
                person_bbox = person.bbox
                t_info = temporal_valid_map.get(track_id, {})
                is_temporally_valid = t_info.get("is_valid", False)
                gun_duration_sec = t_info.get("duration_sec", 0.0)
                
                # Flag person in tracker as suspect / armed
                self.tracker.mark_suspect(track_id, is_armed=True)
            else:
                track_id = f"gun_unassociated_{uuid.uuid4().hex[:4]}"
                person_bbox = None
                is_temporally_valid = True

            crop_sim_info = frame.get("crop_sim_info", {}) if isinstance(frame, dict) else {}
            if isinstance(frame, dict) and "sim_duration_sec" in frame:
                gun_duration_sec = frame["sim_duration_sec"]

            # VECTOR VERIFICATION
            if is_degraded:
                vector_score = 0.5
                matched_category = "degraded_mode_bypass"
            else:
                vector_score, matched_category = self.vector_verify.verify_crop(crop_sim_info)

            # POSE / CONTEXT CHECK
            pose_info = self.context_checker.analyze_pose(frame, person_bbox, gun["bbox"])
            pose_risk_factor = pose_info["pose_risk_factor"]

            track_stability = 0.9

            # RISK SCORING
            metrics = {
                "detection_conf": gun["confidence"],
                "pose_risk": pose_risk_factor,
                "location_risk": location_risk,
                "track_stability": track_stability,
                "vector_score": vector_score
            }
            
            risk_result = self.risk_scorer.compute(metrics)
            
            if gun_duration_sec >= 3.0 or risk_result["score"] >= 0.75:
                risk_result["level"] = "DANGER"
            elif is_degraded or watch_state:
                if risk_result["level"] == "LOW":
                    risk_result["level"] = "MEDIUM"

            dispatch_outcome = self.alert_system.dispatch(track_id, risk_result, frame)

            detections_results.append({
                "track_id": track_id,
                "gun_bbox": gun["bbox"],
                "person_bbox": person_bbox,
                "class": gun.get("class", "gun"),
                "confidence": gun["confidence"],
                "temporal_valid": is_temporally_valid,
                "gun_duration_sec": gun_duration_sec,
                "vector_match": matched_category,
                "vector_score": vector_score,
                "pose_context": pose_info["context"],
                "risk_score": risk_result["score"],
                "risk_level": risk_result["level"],
                "action": dispatch_outcome["action_taken"],
                "evidence_hash": dispatch_outcome.get("evidence_hash", None),
                "degraded_mode": is_degraded
            })

        # 6. CONCEALMENT / STICKY SUSPECT EVALUATION
        # Loop through tracked persons to assemble output payload
        persons_payload = []
        now = time.time()
        for pid, p in tracked_persons.items():
            # Sticky suspect check: if previously armed/suspect but 0 active gun bboxes in current frame
            is_concealed = False
            if p.previously_armed or p.is_suspect:
                # Retention window: keep suspect state active for 300 seconds after weapon disappears
                if p.last_armed_timestamp > 0 and (now - p.last_armed_timestamp) < 300:
                    is_concealed = True

            persons_payload.append({
                "track_id": pid,
                "bbox": p.bbox,
                "is_suspect": p.is_suspect or is_concealed,
                "previously_armed": p.previously_armed,
                "is_concealed": is_concealed
            })

        return {
            "persons": persons_payload,
            "detections": detections_results,
            "quality": q_metrics
        }

    def clear_suspect_status(self, track_id: str):
        """
        Operator audit clearance protocol method.
        """
        self.tracker.clear_suspect(track_id)
        logger.info(f"AUDIT PROTOCOL: Operator cleared suspect status for track ID '{track_id}'.")
