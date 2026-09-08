from typing import Dict
from .config import RiskConfig

class ThreatRiskScorer:
    def __init__(self, config: RiskConfig):
        self.config = config

    # Standard python signature
    def compute(self, metrics: Dict) -> Dict:
        """
        Computes threat risk score based on configured weights.
        metrics: Dict containing:
            - detection_conf (0.0 to 1.0)
            - pose_risk (0.0 to 1.0)
            - location_risk (0.0 to 1.0) - e.g., school lobby (high) vs gun range (low)
            - track_stability (0.0 to 1.0)
            - vector_score (0.0 to 1.0)
        """
        w = self.config.weights
        
        det_part = metrics.get("detection_conf", 0.0) * w.get("detection_conf", 0.35)
        pose_part = metrics.get("pose_risk", 0.0) * w.get("pose_risk", 0.25)
        loc_part = metrics.get("location_risk", 0.0) * w.get("location_risk", 0.10)
        track_part = metrics.get("track_stability", 0.5) * w.get("track_stability", 0.10)
        vec_part = metrics.get("vector_score", 0.0) * w.get("vector_score", 0.20)

        total_score = det_part + pose_part + loc_part + track_part + vec_part
        # Ensure bounds
        total_score = max(0.0, min(1.0, total_score))

        # Risk Classification
        if total_score >= self.config.high_threshold:
            classification = "HIGH"
        elif total_score >= self.config.medium_threshold:
            classification = "MEDIUM"
        else:
            classification = "LOW"

        return {
            "score": total_score,
            "level": classification
        }
