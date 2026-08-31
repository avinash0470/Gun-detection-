from typing import Dict

class PoseContextChecker:
    def __init__(self):
        # Maps categories to risk multiplier
        self.context_risks = {
            "aimed": 1.0,
            "carried_ready": 0.8,
            "slung": 0.3,
            "holstered": 0.1,
            "concealed": 0.6,
            "casual": 0.5,
            "unknown": 0.4
        }

    def analyze_pose(self, frame, person_bbox, gun_bbox) -> Dict:
        """
        Determines the contextual state of the gun relative to the person.
        In production, this hooks up to a Keypoint/Pose estimator (like YOLOv8-pose)
        to check if the hand overlaps the gun handle and the arm is extended (aimed).
        """
        # Mock analysis
        context = "unknown"
        if isinstance(frame, dict):
            context = frame.get("pose_context", "unknown")

        risk_score = self.context_risks.get(context, 0.4)

        return {
            "context": context,
            "pose_risk_factor": risk_score
        }
