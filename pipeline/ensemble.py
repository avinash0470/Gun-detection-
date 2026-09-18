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
    def __init__(self, config: DetectorConfig, hardware_profile=None):
        self.config = config
        self.hardware = hardware_profile
        self.model = None

        # Determine device string and imgsz
        if self.hardware:
            self.device = self.hardware.device_str
            self.imgsz = self.config.imgsz if self.config.imgsz > 0 else self.hardware.imgsz
            self.half = self.hardware.use_half if self.config.half is None else self.config.half
        else:
            self.device = getattr(config, "device", "auto")
            self.imgsz = 480
            self.half = False

        # Load firearm detection model: exclusively best_gun_26n(23).pt
        gun_model_filename = "best_gun_26n(23).pt"
        custom_model = getattr(self.config, "gun_model", None) or gun_model_filename
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        candidate_paths = [
            custom_model,
            os.path.join(project_root, os.path.basename(custom_model)),
            os.path.join(project_root, gun_model_filename),
            os.path.join(os.getcwd(), gun_model_filename),
            gun_model_filename,
        ]

        selected_model_path = None
        for path in candidate_paths:
            if path and os.path.exists(path):
                selected_model_path = os.path.abspath(path)
                break

        if ULTRALYTICS_AVAILABLE:
            if selected_model_path:
                try:
                    self.model = YOLO(selected_model_path)
                    logger.info(f"Loaded firearm YOLO model from '{selected_model_path}' on device={self.device} (imgsz={self.imgsz}, half={self.half}).")
                except Exception as e:
                    logger.error(f"Failed to load YOLO model from '{selected_model_path}': {e}")
            else:
                logger.warning(f"Firearm model '{gun_model_filename}' not found. Running in mock/simulation mode.")
        else:
            logger.warning("ultralytics package not installed. Primary detector running in mock mode.")

    def detect(self, frame, threshold_offset: float = 0.0) -> Tuple[List[Dict], bool]:
        """
        Runs primary (YOLO best_gun.pt) and secondary detectors on the frame.
        Applies a threshold offset calculated by the Quality Gate.
        """
        primary_threshold = max(0.05, self.config.primary_conf_threshold + threshold_offset)
        secondary_threshold = max(0.05, self.config.secondary_conf_threshold + threshold_offset)

        primary_raw = []
        secondary_raw = []

        # 1. Run real model inference if available and frame is not a dictionary (mock)
        if self.model and not isinstance(frame, dict):
            try:
                pred_kwargs = {
                    "conf": primary_threshold,
                    "iou": 0.40,
                    "imgsz": self.imgsz,
                    "verbose": False
                }
                if self.device != "auto":
                    pred_kwargs["device"] = self.device
                if self.half and self.device != "cpu":
                    pred_kwargs["half"] = True

                results = self.model(frame, **pred_kwargs)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
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
                # Disagreement! Primary detected, secondary didn't (only when secondary detector is active)
                if len(secondary_passed) > 0 or isinstance(frame, dict):
                    watch_state = True
                confirmed_candidates.append({
                    "bbox": p_bbox,
                    "confidence": p_det["confidence"],
                    "class": p_det.get("class", "gun"),
                    "detector": "primary_yolo"
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
