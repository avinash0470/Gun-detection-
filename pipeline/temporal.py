import time
from collections import deque
from typing import Dict, List
from .config import TemporalConfig

class TemporalValidator:
    def __init__(self, config: TemporalConfig):
        self.config = config
        self.track_history: Dict[str, deque] = {}
        self.occlusion_counters: Dict[str, int] = {}
        # Frames since track was last seen in active_track_ids
        self.absent_counters: Dict[str, int] = {}
        # Timestamps for tracking long exposure to gun threat
        self.gun_first_seen_timestamp: Dict[str, float] = {}

    def validate(self, active_track_ids: List[str], current_detections: List[str]) -> Dict[str, Dict]:
        """
        Validates detections across time and tracks persistent firearm exposure duration.
        Returns:
            Dict[str, Dict]: track_id -> {"is_valid": bool, "duration_sec": float}
        """
        results = {}
        now = time.time()
        active_set = set(active_track_ids)

        for track_id in active_track_ids:
            self.absent_counters.pop(track_id, None)

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.config.history_frames)
                self.occlusion_counters[track_id] = 0

            is_detected = track_id in current_detections
            self.track_history[track_id].append(is_detected)

            if is_detected:
                self.occlusion_counters[track_id] = 0
                if track_id not in self.gun_first_seen_timestamp:
                    self.gun_first_seen_timestamp[track_id] = now
            else:
                self.occlusion_counters[track_id] += 1
                # If gun disappears for 3 consecutive frames, reset exposure timer
                if self.occlusion_counters[track_id] >= 3:
                    self.gun_first_seen_timestamp.pop(track_id, None)

            detections_count = sum(self.track_history[track_id])
            is_valid = detections_count >= self.config.min_detections_in_window

            # Duration only accumulates when weapon detection is actively validated across multiple frames
            duration_sec = 0.0
            if is_valid and track_id in self.gun_first_seen_timestamp:
                duration_sec = max(0.0, now - self.gun_first_seen_timestamp[track_id])

            results[track_id] = {
                "is_valid": is_valid,
                "duration_sec": duration_sec
            }

        # Age out tracks completely absent from active_track_ids independent of gun occlusion
        for dt in list(self.track_history.keys()):
            if dt not in active_set:
                self.absent_counters[dt] = self.absent_counters.get(dt, 0) + 1
                if self.absent_counters[dt] > self.config.occlusion_recovery_frames:
                    del self.track_history[dt]
                    self.occlusion_counters.pop(dt, None)
                    self.gun_first_seen_timestamp.pop(dt, None)
                    del self.absent_counters[dt]
            else:
                self.absent_counters.pop(dt, None)

        return results
