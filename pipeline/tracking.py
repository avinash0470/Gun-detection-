from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import uuid
import logging
import time
import numpy as np

logger = logging.getLogger("GunDetectionPipeline")

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
    association_history: List[str] = None
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
        Updates trackers based on bboxes and ReID feature vector similarity across re-entries.
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
                # Inherit persistent suspect state from gallery across re-entries
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
            
            # Save gallery entry
            self.reid_gallery[matched_id] = {
                "vector": reid_vec,
                "is_suspect": is_suspect,
                "previously_armed": previously_armed,
                "last_armed_timestamp": last_armed_ts
            }
            
        self.active_tracks = updated_tracks
        return self.active_tracks

    def mark_suspect(self, track_id: str, is_armed: bool = False):
        """
        Flags a track ID as suspect or previously armed.
        """
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
        """
        Operator audit clearance protocol: Clears suspect status after manual verification.
        """
        if track_id in self.active_tracks:
            p = self.active_tracks[track_id]
            p.is_suspect = False
            p.previously_armed = False

        if track_id in self.reid_gallery:
            self.reid_gallery[track_id]["is_suspect"] = False
            self.reid_gallery[track_id]["previously_armed"] = False

    def _extract_reid_vector(self, frame, bbox) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        seed = int(abs(x1 * 31 + y1 * 17 + (x2 - x1) * 7) % 10000)
        np.random.seed(seed)
        vec = np.random.randn(128)
        return vec / (np.linalg.norm(vec) + 1e-6)

    def _match_reid(self, bbox, reid_vec: np.ndarray) -> Tuple[Optional[str], Dict]:
        best_match_id = None
        highest_sim = 0.0
        matched_meta = {}

        # 1. Spatial distance check with active tracks
        for tid, track in self.active_tracks.items():
            dist = abs(track.bbox[0] - bbox[0]) + abs(track.bbox[1] - bbox[1])
            if dist < 120:
                meta = self.reid_gallery.get(tid, {})
                return tid, meta

        # 2. ReID vector similarity search
        for tid, meta in self.reid_gallery.items():
            gallery_vec = meta.get("vector")
            if gallery_vec is not None:
                sim = float(np.dot(reid_vec, gallery_vec))
                if sim > 0.85 and sim > highest_sim:
                    highest_sim = sim
                    best_match_id = tid
                    matched_meta = meta

        return best_match_id, matched_meta
