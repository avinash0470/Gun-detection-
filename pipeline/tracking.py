from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import uuid
import logging
import time
import numpy as np

logger = logging.getLogger("GunDetectionPipeline")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

@dataclass
class TrackedPerson:
    track_id: str
    bbox: Tuple[int, int, int, int]
    reid_vector: np.ndarray
    association_history: List[str] = field(default_factory=list)
    is_suspect: bool = False
    previously_armed: bool = False
    last_armed_timestamp: float = 0.0

class PersonTracker:
    def __init__(self, config=None, hardware_profile=None):
        self.config = config
        self.hardware = hardware_profile
        
        if self.hardware:
            self.device = self.hardware.device_str
            self.imgsz = self.config.imgsz if (self.config and self.config.imgsz > 0) else self.hardware.imgsz
            self.half = self.hardware.use_half if (self.config and self.config.half is None) else (getattr(self.config, "half", False) or False)
        else:
            self.device = getattr(config, "device", "auto") if config else "auto"
            self.imgsz = 480
            self.half = False

        self.active_tracks: Dict[str, TrackedPerson] = {}
        self.reid_gallery: Dict[str, Dict] = {} # Persistent gallery with suspect metadata
        self.person_model = None
        
        if ULTRALYTICS_AVAILABLE:
            try:
                self.person_model = YOLO("yolo11m.pt")
                logger.info(f"Loaded person tracking model 'yolo11m.pt' on device={self.device} (imgsz={self.imgsz}).")
            except Exception as e:
                logger.warning(f"Could not load yolo11m.pt for person tracking: {e}")

    def update(self, frame, detections: List[Dict]) -> Dict[str, TrackedPerson]:
        """
        Updates trackers based on bboxes, native ByteTrack trajectory tracking, and visual appearance ReID vectors.
        """
        if not isinstance(frame, dict) and self.person_model and not detections:
            try:
                # Use ByteTrack with configured device and resolution
                track_kwargs = {
                    "persist": True, 
                    "tracker": "bytetrack.yaml", 
                    "classes": [0], 
                    "conf": 0.25, 
                    "imgsz": self.imgsz, 
                    "verbose": False
                }
                if self.device != "auto":
                    track_kwargs["device"] = self.device
                if self.half and self.device != "cpu":
                    track_kwargs["half"] = True

                results = self.person_model.track(frame, **track_kwargs)
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            track_id = f"person_{int(box.id[0].item())}" if (box.id is not None and len(box.id) > 0) else None
                            detections.append({"bbox": (x1, y1, x2, y2), "track_id": track_id})
            except Exception as e:
                # Fallback to standard detect if tracking backend has issue
                try:
                    pred_kwargs = {"classes": [0], "conf": 0.35, "verbose": False}
                    if self.device != "auto":
                        pred_kwargs["device"] = self.device
                    results = self.person_model(frame, **pred_kwargs)
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            detections.append({"bbox": (x1, y1, x2, y2)})
                except Exception as e2:
                    logger.error(f"Error during person detection: {e2}")

        updated_tracks = {}
        for det in detections:
            bbox = det.get("bbox", (0, 0, 100, 200))
            reid_vec = self._extract_reid_vector(frame, bbox)
            
            # Step 1: Check if the person matches an existing person in the persistent ReID gallery
            matched_id, gallery_meta = self._match_reid(bbox, reid_vec)
            
            # Step 2: If no gallery match, try YOLO ByteTrack ID or assign consistent new ID
            if matched_id is None:
                provided_id = det.get("track_id")
                if provided_id and provided_id not in self.reid_gallery:
                    matched_id = provided_id
                else:
                    matched_id = f"person_{len(self.reid_gallery) + 1}"
                is_suspect = False
                previously_armed = False
                last_armed_ts = 0.0
            else:
                is_suspect = gallery_meta.get("is_suspect", False)
                previously_armed = gallery_meta.get("previously_armed", False)
                last_armed_ts = gallery_meta.get("last_armed_timestamp", 0.0)

            tracked_person = TrackedPerson(
                track_id=matched_id,
                bbox=bbox,
                reid_vector=reid_vec,
                association_history=[],
                is_suspect=is_suspect,
                previously_armed=previously_armed,
                last_armed_timestamp=last_armed_ts
            )
            
            updated_tracks[matched_id] = tracked_person
            
            # Update Gallery with moving average appearance vector for persistent tracking
            if reid_vec is not None:
                if matched_id in self.reid_gallery and self.reid_gallery[matched_id].get("vector") is not None:
                    old_v = self.reid_gallery[matched_id]["vector"]
                    # Smooth visual appearance update (exponential moving average)
                    updated_v = (old_v * 0.7) + (reid_vec * 0.3)
                    norm = np.linalg.norm(updated_v)
                    if norm > 1e-6:
                        updated_v = updated_v / norm
                else:
                    updated_v = reid_vec

                self.reid_gallery[matched_id] = {
                    "vector": updated_v,
                    "last_bbox": bbox,
                    "is_suspect": is_suspect,
                    "previously_armed": previously_armed,
                    "last_armed_timestamp": last_armed_ts
                }
            
        self.active_tracks = updated_tracks
        return self.active_tracks

    def mark_suspect(self, track_id: str, is_armed: bool = False):
        now = time.time()
        if track_id in self.active_tracks:
            p = self.active_tracks[track_id]
            p.is_suspect = True
            if is_armed:
                p.previously_armed = True
                p.last_armed_timestamp = now

        if track_id in self.reid_gallery:
            self.reid_gallery[track_id]["is_suspect"] = True
            if is_armed:
                self.reid_gallery[track_id]["previously_armed"] = True
                self.reid_gallery[track_id]["last_armed_timestamp"] = now

    def clear_suspect(self, track_id: str):
        if track_id in self.active_tracks:
            p = self.active_tracks[track_id]
            p.is_suspect = False
            p.previously_armed = False

        if track_id in self.reid_gallery:
            self.reid_gallery[track_id]["is_suspect"] = False
            self.reid_gallery[track_id]["previously_armed"] = False

    def _extract_reid_vector(self, frame, bbox) -> Optional[np.ndarray]:
        """
        Extracts a multi-part visual appearance feature vector:
        1. Upper body (torso/clothing color histogram)
        2. Lower body (pants/lower clothing color histogram)
        3. Full-body color & texture signature
        Enables persistent cross-camera & re-entry identity matching.
        """
        if not isinstance(frame, dict) and OPENCV_AVAILABLE and frame is not None and hasattr(frame, "shape"):
            try:
                x1, y1, x2, y2 = bbox
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if (x2 - x1) > 15 and (y2 - y1) > 25:
                    crop = frame[y1:y2, x1:x2]
                    h_crop, w_crop = crop.shape[:2]
                    
                    # Split into Upper Body (torso) and Lower Body (legs)
                    mid_y = int(h_crop * 0.5)
                    upper_crop = crop[:mid_y, :]
                    lower_crop = crop[mid_y:, :]
                    
                    # Compute HSV histograms for Upper and Lower portions
                    upper_hsv = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2HSV)
                    lower_hsv = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2HSV)
                    
                    hist_upper = cv2.calcHist([upper_hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
                    hist_lower = cv2.calcHist([lower_hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
                    
                    vec_upper = cv2.normalize(hist_upper, hist_upper).flatten()
                    vec_lower = cv2.normalize(hist_lower, hist_lower).flatten()
                    
                    full_vector = np.concatenate([vec_upper, vec_lower])
                    norm = np.linalg.norm(full_vector)
                    if norm > 1e-6:
                        full_vector = full_vector / norm
                    return full_vector
            except Exception:
                pass

        return None

    def _match_reid(self, bbox, reid_vec: Optional[np.ndarray]) -> Tuple[Optional[str], Dict]:
        """
        Robust ReID matching across frames and cameras:
        1. Continuous Track: High IoU (> 0.25) with recently active tracks.
        2. Re-entry & Camera Change: Visual ReID cosine similarity (> 0.70) against persistent gallery.
        """
        best_match_id = None
        highest_iou = 0.0

        # 1. IoU Trajectory Check against active tracks
        for tid, track in self.active_tracks.items():
            iou = self._compute_iou(bbox, track.bbox)
            if iou > 0.25 and iou > highest_iou:
                highest_iou = iou
                best_match_id = tid

        if best_match_id:
            meta = self.reid_gallery.get(best_match_id, {})
            return best_match_id, meta

        # 2. Visual ReID appearance similarity against the entire persistent gallery
        if reid_vec is not None:
            highest_sim = 0.0
            matched_gallery_id = None
            matched_meta = {}

            for tid, meta in self.reid_gallery.items():
                gallery_vec = meta.get("vector")
                if gallery_vec is not None and len(gallery_vec) == len(reid_vec):
                    sim = float(np.dot(reid_vec, gallery_vec))
                    # Appearance similarity threshold for cross-camera / re-entry matching
                    if sim > 0.68 and sim > highest_sim:
                        highest_sim = sim
                        matched_gallery_id = tid
                        matched_meta = meta

            if matched_gallery_id:
                return matched_gallery_id, matched_meta

        return None, {}

    def _compute_iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
