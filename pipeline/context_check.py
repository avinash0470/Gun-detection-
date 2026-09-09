from typing import Dict, Any, Tuple, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class PoseContextChecker:
    def __init__(self, hardware_profile=None):
        self.hardware = hardware_profile
        self.pose_model = None
        self.enable_pose_keypoints = getattr(hardware_profile, "enable_pose_keypoints", False) if hardware_profile else False
        self.device = getattr(hardware_profile, "device_str", "cpu") if hardware_profile else "cpu"

        # Only load pose neural network if GPU is present (to save CPU cycles and prevent stutter)
        if ULTRALYTICS_AVAILABLE and self.enable_pose_keypoints:
            try:
                self.pose_model = YOLO("yolo11m-pose.pt")
            except Exception:
                self.pose_model = None

        # Maps categories to risk multiplier
        self.context_risks = {
            "aimed": 1.0,
            "carried_ready": 0.8,
            "in_hand": 0.9,
            "slung": 0.3,
            "holstered": 0.1,
            "concealed": 0.6,
            "casual": 0.5,
            "unknown": 0.4
        }

    def analyze_pose(self, frame, person_bbox, gun_bbox) -> Dict:
        """
        Determines if the gun is in/near the person's hand or wrist with high speed and motion tolerance.
        """
        context = "unknown"
        is_in_hand = False
        wrist_dist_min = 9999.0

        if isinstance(frame, dict):
            context = frame.get("pose_context", "unknown")
            is_in_hand = (context in ["aimed", "carried_ready", "in_hand"])
            risk_score = self.context_risks.get(context, 0.4)
            return {
                "context": context,
                "pose_risk_factor": risk_score,
                "is_in_hand": is_in_hand,
                "wrist_distance": 0.0 if is_in_hand else 999.0
            }

        if person_bbox and gun_bbox:
            px1, py1, px2, py2 = person_bbox
            gx1, gy1, gx2, gy2 = gun_bbox
            g_cx = (gx1 + gx2) / 2.0
            g_cy = (gy1 + gy2) / 2.0
            p_w = max(1, px2 - px1)
            p_h = max(1, py2 - py1)

            # 1. Fast Spatial / Arm-Reach Heuristic (Zero CPU inference cost, handles fast motion blur & perspective)
            # Firearms held in hand are within the person's bounding box expanded by arm-reach (50% width)
            # and between 10% (shoulders) to 110% (hands/hips/low-reach) of vertical person height
            in_reach_x = (px1 - p_w * 0.50) <= g_cx <= (px2 + p_w * 0.50)
            in_reach_y = (py1 + p_h * 0.08) <= g_cy <= (py2 + p_h * 0.18)

            if in_reach_x and in_reach_y:
                is_in_hand = True
                context = "in_hand"

            # 2. Keypoint check for refined wrist distance if pose model is enabled on GPU
            if self.pose_model and frame is not None and hasattr(frame, "shape") and not is_in_hand:
                try:
                    h_img, w_img = frame.shape[:2]
                    px1_c = max(0, int(px1 - 30))
                    py1_c = max(0, int(py1 - 30))
                    px2_c = min(w_img, int(px2 + 30))
                    py2_c = min(h_img, int(py2 + 30))

                    person_crop = frame[py1_c:py2_c, px1_c:px2_c]
                    if person_crop.size > 0:
                        pose_kwargs = {"conf": 0.20, "imgsz": 320, "verbose": False}
                        if self.device != "auto":
                            pose_kwargs["device"] = self.device
                        results = self.pose_model(person_crop, **pose_kwargs)
                        for res in results:
                            if res.keypoints is not None and len(res.keypoints.xy) > 0:
                                kpts = res.keypoints.xy[0].cpu().numpy()
                                for kpt_idx in [9, 10]: # Left & Right wrists
                                    if kpt_idx < len(kpts):
                                        kx_local, ky_local = kpts[kpt_idx]
                                        if kx_local > 0 and ky_local > 0:
                                            kx = kx_local + px1_c
                                            ky = ky_local + py1_c
                                            d = ((kx - g_cx)**2 + (ky - g_cy)**2)**0.5
                                            if d < wrist_dist_min:
                                                wrist_dist_min = d
                                            if d <= max(100.0, p_w * 0.8):
                                                is_in_hand = True
                                                context = "in_hand"
                except Exception:
                    pass

        risk_score = self.context_risks.get(context, 0.8 if is_in_hand else 0.2)

        return {
            "context": context,
            "pose_risk_factor": risk_score,
            "is_in_hand": is_in_hand,
            "wrist_distance": wrist_dist_min
        }
