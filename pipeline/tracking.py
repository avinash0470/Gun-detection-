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
    def __init__(self):
        self.active_tracks: Dict[str, TrackedPerson] = {}
        self.reid_gallery: Dict[str, Dict] = {} # Persistent gallery with suspect metadata
        self.person_model = None
        
        if ULTRALYTICS_AVAILABLE:
            try:
                self.person_model = YOLO("yolo11m.pt")
                logger.info("Loaded person tracking model 'yolo11m.pt' successfully.")
            except Exception as e:
                logger.warning(f"Could not load yolo11m.pt for person tracking: {e}")

    def update(self, frame, detections: List[Dict]) -> Dict[str, TrackedPerson]:
        """
        Updates trackers based on bboxes, IoU trajectory tracking, and visual appearance ReID vectors.
        """
        if not isinstance(frame, dict) and self.person_model and not detections:
            try:
                results = self.person_model(frame, classes=[0], conf=0.3, verbose=False)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detections.append({"bbox": (x1, y1, x2, y2)})
            except Exception as e:
                logger.error(f"Error during person detection: {e}")

        updated_tracks = {}
        for det in detections:
            bbox = det.get("bbox", (0, 0, 100, 200))
            reid_vec = self._extract_reid_vector(frame, bbox)
            
            matched_id, gallery_meta = self._match_reid(bbox, reid_vec)
            
            if matched_id is None:
                matched_id = f"person_{uuid.uuid4().hex[:6]}"
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
            
            self.reid_gallery[matched_id] = {
                "vector": reid_vec,
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

    def _extract_reid_vector(self, frame, bbox) -> np.ndarray:
        """
        Extracts stable visual appearance feature vector (HSV Color Histogram).
        This avoids random seed coordinate mutation bugs.
        """
        if not isinstance(frame, dict) and OPENCV_AVAILABLE and frame is not None:
            try:
                x1, y1, x2, y2 = bbox
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    crop = frame[y1:y2, x1:x2]
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    # Compute 32-bin Hue & Saturation color histogram for person appearance
                    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
                    vec = cv2.normalize(hist, hist).flatten()
                    return vec
            except Exception as e:
                pass

        # Stable fallback vector based on fixed dimension ratio
        x1, y1, x2, y2 = bbox
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        aspect = float(h) / float(w)
        vec = np.ones(128) * (aspect * 0.1)
        return vec / (np.linalg.norm(vec) + 1e-6)

    def _match_reid(self, bbox, reid_vec: np.ndarray) -> Tuple[Optional[str], Dict]:
        """
        Matching strategy:
        1. Intersection over Union (IoU) / Proximity with active tracks.
        2. Visual ReID cosine similarity check ONLY if IoU fails.
        """
        best_match_id = None
        highest_iou = 0.0

        # 1. IoU Trajectory Check against active tracks
        for tid, track in self.active_tracks.items():
            iou = self._compute_iou(bbox, track.bbox)
            if iou > 0.3 and iou > highest_iou:
                highest_iou = iou
                best_match_id = tid

        if best_match_id:
            meta = self.reid_gallery.get(best_match_id, {})
            return best_match_id, meta

        # 2. Visual ReID appearance similarity against suspect gallery (for re-entry)
        highest_sim = 0.0
        matched_meta = {}
        for tid, meta in self.reid_gallery.items():
            # Only match against gallery entries that were marked SUSPECT/ARMED
            if not meta.get("is_suspect", False) and not meta.get("previously_armed", False):
                continue

            gallery_vec = meta.get("vector")
            if gallery_vec is not None and len(gallery_vec) == len(reid_vec):
                sim = float(np.dot(reid_vec, gallery_vec))
                if sim > 0.90 and sim > highest_sim:
                    highest_sim = sim
                    best_match_id = tid
                    matched_meta = meta

        return best_match_id, matched_meta

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
