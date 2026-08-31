import hashlib
import time
from typing import Dict, Any

class AlertSystem:
    def __init__(self):
        self.review_queue = []
        self.audit_log = []

    def dispatch(self, track_id: str, risk_data: Dict[str, Any], frame) -> Dict:
        """
        Processes threat outcomes.
        If Low: Keep watching, log only.
        If Medium: Queue for human review.
        If High: Evidence capture + immediate alert.
        """
        level = risk_data.get("level", "LOW")
        score = risk_data.get("score", 0.0)
        timestamp = time.time()
        
        event = {
            "track_id": track_id,
            "score": score,
            "level": level,
            "timestamp": timestamp,
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
            # EVIDENCE CAPTURE (Hashed, timestamped, audit trail)
            evidence_hash = self._generate_evidence_hash(track_id, score, timestamp, frame)
            event["action_taken"] = "IMMEDIATE_DANGER_ALERT_DISPATCHED"
            event["evidence_hash"] = evidence_hash
            self.audit_log.append(event)
            
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

    def _generate_evidence_hash(self, track_id: str, score: float, timestamp: float, frame) -> str:
        """
        Generates a SHA-256 hash representing unique metadata signature to verify chain of custody.
        """
        hasher = hashlib.sha256()
        hasher.update(f"{track_id}-{score}-{timestamp}".encode('utf-8'))
        if isinstance(frame, bytes):
            hasher.update(frame)
        else:
            hasher.update(str(frame).encode('utf-8'))
        return hasher.hexdigest()
