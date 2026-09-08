import numpy as np
from .config import QualityGateConfig

class FrameQualityGate:
    def __init__(self, config: QualityGateConfig):
        self.config = config

    def process(self, frame) -> dict:
        """
        Analyzes the frame and returns quality metrics.
        In a production system, this calculates laplacian variance (blur)
        and mean pixel intensities (brightness).
        """
        # Mock calculation: if frame is a dict/simulated, get properties, else analyze numpy array
        if isinstance(frame, dict):
            brightness = frame.get("brightness", 0.8)
            blur_metric = frame.get("blur", 150.0)
            occlusion_score = frame.get("occlusion", 0.0)
        else:
            # Real frame analysis using numpy arrays
            brightness = float(np.mean(frame)) / 255.0 if frame is not None else 0.8
            # Blur detection using Variance of Laplacian (low value = blurry)
            try:
                import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_metric = cv2.Laplacian(gray, cv2.CV_64F).var()
            except Exception:
                blur_metric = 120.0
            occlusion_score = 0.0

        is_low_light = brightness < self.config.low_light_threshold
        is_blurry = blur_metric < self.config.blur_threshold
        is_occluded = occlusion_score > self.config.occlusion_threshold

        # Compute dynamic threshold adjustments
        # Cap penalties so motion blur does not block detection of moving firearms
        threshold_offset = 0.0
        if is_low_light:
            threshold_offset += 0.05
        if is_blurry:
            threshold_offset += 0.05

        return {
            "brightness": brightness,
            "blur_metric": blur_metric,
            "occlusion_score": occlusion_score,
            "is_low_light": is_low_light,
            "is_blurry": is_blurry,
            "is_occluded": is_occluded,
            "threshold_offset": threshold_offset
        }
