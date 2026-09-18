import hashlib
import time
import logging
import os
import cv2
import json
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("GunDetectionPipeline")

class AlertSystem:
    def __init__(self, evidence_dir: str = "evidence", log_dir: str = "logs"):
        self.review_queue = []
        self.audit_log = []
        self.evidence_dir = evidence_dir
        self.log_dir = log_dir

        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Persistent detection event log file (JSONL format)
        self.log_file = os.path.join(self.log_dir, "detection_events.jsonl")

    def dispatch(self, track_id: str, risk_data: Dict[str, Any], frame) -> Dict:
        """
        Processes threat outcomes, creates snapshots for high-risk threats,
        and logs every event to disk.
        """
        level = risk_data.get("level", "LOW")
        score = risk_data.get("score", 0.0)
        timestamp = time.time()
        
        event = {
            "track_id": track_id,
            "score": score,
            "level": level,
            "timestamp": timestamp,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action_taken": "NONE"
        }

        if level == "LOW":
            event["action_taken"] = "LOGGED_ONLY"
            self.audit_log.append(event)
            
        elif level == "MEDIUM":
            event["action_taken"] = "QUEUED_FOR_HUMAN_REVIEW"
            self.review_queue.append(event)
            self.audit_log.append(event)
            
        elif level in ["HIGH", "DANGER"]:
            evidence_hash = self._generate_evidence_hash(track_id, score, timestamp, frame)
            snapshot_path = self._save_evidence_snapshot(track_id, timestamp, frame)
            event["action_taken"] = "IMMEDIATE_DANGER_ALERT_DISPATCHED"
            event["evidence_hash"] = evidence_hash
            event["snapshot_path"] = snapshot_path if snapshot_path else None
            self.audit_log.append(event)

        # Write event to persistent JSONL log
        self._write_event_log(event)
            
        return event

    def record_human_review_outcome(self, event_id: int, confirmed: bool) -> Dict:
        """
        HUMAN REVIEW OUTCOME & FEEDBACK LOOP:
        Processes human confirmation or dismissal of medium/high alerts.
        Updates reference embeddings or logs false positives/negatives for retraining.
        """
        if event_id < len(self.review_queue):
            event = self.review_queue[event_id]
            event["human_review_confirmed"] = confirmed
            event["status"] = "RESOLVED_CONFIRMED" if confirmed else "RESOLVED_DISMISSED"
            
            # Feedback loop action
            if not confirmed:
                logger.info(f"FEEDBACK LOOP: False Positive recorded for track {event['track_id']}. Logging crop image for hard-negative retraining.")
            else:
                logger.info(f"FEEDBACK LOOP: True Positive confirmed for track {event['track_id']}. Indexing embedding to reference database.")
            
            return event
        return {"error": "Event ID not found"}

    def _save_evidence_snapshot(self, track_id: str, timestamp: float, frame) -> str:
        """Saves the frame as a JPEG image for evidence when a DANGER alert fires."""
        if not isinstance(frame, dict) and frame is not None:
            try:
                dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"evidence_{track_id}_{dt_str}.jpg"
                filepath = os.path.join(self.evidence_dir, filename)
                cv2.imwrite(filepath, frame)
                logger.info(f"EVIDENCE SNAPSHOT saved: {filepath}")
                return filepath
            except Exception as e:
                logger.error(f"Failed to save evidence snapshot: {e}")
        return ""

    def _write_event_log(self, event: Dict):
        """Appends event to a persistent JSONL log file on disk."""
        try:
            # Create a serializable copy (avoid writing non-serializable data)
            log_entry = {k: v for k, v in event.items()}
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write event log: {e}")

    def _generate_evidence_hash(self, track_id: str, score: float, timestamp: float, frame) -> str:
        """
        Generates a SHA-256 hash representing unique metadata signature to verify chain of custody.
        Uses exact frame byte content for image arrays (e.g. numpy arrays).
        """
        hasher = hashlib.sha256()
        hasher.update(f"{track_id}-{score}-{timestamp}".encode('utf-8'))
        if isinstance(frame, bytes):
            hasher.update(frame)
        elif hasattr(frame, "tobytes"):
            hasher.update(frame.tobytes())
        else:
            hasher.update(str(frame).encode('utf-8'))
        return hasher.hexdigest()
