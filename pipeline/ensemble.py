import os
import logging
from typing import List, Dict, Tuple
from .config import DetectorConfig

logger = logging.getLogger("GunDetectionPipeline")

# Try to import ultralytics for loading the real .pt model
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class DetectionEnsemble:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = None

        # Load the best available firearm detection model (best_gun_11m.pt > best_gun.pt)
        candidate_models = ["best_gun_11m.pt", "best_gun.pt"]
        selected_model_path = None
        for path in candidate_models:
            if os.path.exists(path):
                selected_model_path = path
                break

        if ULTRALYTICS_AVAILABLE:
            if selected_model_path:
                try:
                    self.model = YOLO(selected_model_path)
                    logger.info(f"Loaded primary firearm YOLO model from '{selected_model_path}' successfully.")
                except Exception as e:
                    logger.error(f"Failed to load YOLO model from '{selected_model_path}': {e}")
            else:
                logger.warning(f"No firearm model file found (checked {candidate_models}). Running in mock/simulation mode.")
        else:
            logger.warning("ultralytics package not installed. Primary detector running in mock mode. Run 'pip install ultralytics' to enable real YOLO predictions.")

    def detect(self, frame, threshold_offset: float = 0.0) -> Tuple[List[Dict], bool]:
        """
        Runs primary (YOLO best_gun.pt) and secondary detectors on the frame.
        Applies a threshold offset calculated by the Quality Gate.
        
        Returns:
            List[Dict]: Confirmed candidate bounding boxes with confidence.
            bool: Watch state flag indicating detector disagreement.
        """
        primary_threshold = max(0.05, self.config.primary_conf_threshold + threshold_offset)
        secondary_threshold = max(0.05, self.config.secondary_conf_threshold + threshold_offset)

        primary_raw = []
        secondary_raw = []

        # 1. Run real model inference if available and frame is not a dictionary (mock)
        if self.model and not isinstance(frame, dict):
            try:
                results = self.model(frame, conf=primary_threshold, verbose=False)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Extract xyxy bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0].item())
                        cls = int(box.cls[0].item())
                        class_name = result.names[cls] if cls in result.names else str(cls)

                        primary_raw.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf,
                            "class": class_name,
                            "detector": "yolo_real"
                        })
            except Exception as e:
                logger.error(f"Error during YOLO model inference: {e}")

        # 2. Support simulator inputs for main.py checks
        elif isinstance(frame, dict):
            primary_raw = frame.get("detections_primary", [])
            secondary_raw = frame.get("detections_secondary", [])

        # Filter detections by dynamic thresholds
        primary_passed = [d for d in primary_raw if d["confidence"] >= primary_threshold]
        secondary_passed = [d for d in secondary_raw if d["confidence"] >= secondary_threshold]

        confirmed_candidates = []
        watch_state = False

        # Match detection candidates spatially
        primary_matched = set()
        secondary_matched = set()

        for p_idx, p_det in enumerate(primary_passed):
            p_bbox = p_det["bbox"]
            matched_sec = None
            for s_idx, s_det in enumerate(secondary_passed):
                s_bbox = s_det["bbox"]
                # Spatial distance match (simple proximity check)
                dist = abs(p_bbox[0] - s_bbox[0]) + abs(p_bbox[1] - s_bbox[1])
                if dist < 50:
                    matched_sec = s_det
                    secondary_matched.add(s_idx)
                    break
            
            if matched_sec:
                primary_matched.add(p_idx)
                avg_conf = (p_det["confidence"] + matched_sec["confidence"]) / 2.0
                confirmed_candidates.append({
                    "bbox": p_bbox,
                    "confidence": avg_conf,
                    "class": p_det.get("class", "gun"),
                    "detector": "ensemble_agreement"
                })
            else:
                # Disagreement! Primary detected, secondary didn't
                watch_state = True
                confirmed_candidates.append({
                    "bbox": p_bbox,
                    "confidence": p_det["confidence"],
                    "class": p_det.get("class", "gun"),
                    "detector": "primary_only_watch"
                })

        # Check secondary detections not matched to primary
        for s_idx, s_det in enumerate(secondary_passed):
            if s_idx not in secondary_matched:
                watch_state = True
                confirmed_candidates.append({
                    "bbox": s_det["bbox"],
                    "confidence": s_det["confidence"],
                    "class": s_det.get("class", "gun"),
                    "detector": "secondary_only_watch"
                })

        return confirmed_candidates, watch_state
