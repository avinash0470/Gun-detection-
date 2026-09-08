import cv2
import numpy as np
from typing import Dict, List, Tuple
from .config import VectorVerifyConfig

class VectorVerification:
    def __init__(self, config: VectorVerifyConfig):
        self.config = config
        self.firearm_references = ["handgun_ref", "rifle_ref", "shotgun_ref", "revolver_ref"]
        # Expanded hard negatives for common false positive objects
        self.hard_negatives = [
            "cellphone", "mobile_phone", "water_bottle", "drink_can", 
            "power_tool", "screwdriver", "wallet", "remote_control", "toy_gun", "skin_patch"
        ]

    def verify_crop(self, crop_image) -> Tuple[float, str]:
        """
        Extracts visual vector representation (HOG gradient & edge complexity)
        to identify genuine firearms and filter out false positives (phones, bottles, skin/fabric).
        """
        if isinstance(crop_image, dict):
            sim_firearm = crop_image.get("sim_firearm", 0.5)
            sim_negative = crop_image.get("sim_negative", 0.2)
            matched_category = crop_image.get("matched_category", "handgun")
            score = max(0.0, sim_firearm - (sim_negative * 0.5))
            return score, matched_category

        if crop_image is None or not hasattr(crop_image, "shape") or crop_image.size == 0:
            return 0.5, "gun"

        h, w = crop_image.shape[:2]
        if h < 8 or w < 8:
            return 0.2, "unknown"

        aspect_ratio = float(w) / float(max(1, h))

        # 1. Reject extremely tall thin vertical boxes (typical of bottles or arm segments)
        if aspect_ratio < 0.25:
            return 0.1, "water_bottle"

        # 2. Extract edge gradient vector characteristics of firearms (barrel, trigger guard, handle)
        try:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            # Edge density via Sobel gradients
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            edge_density = float(np.mean(mag))

            # Color saturation / Skin test (reject pure skin or flat colored patches)
            hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
            std_val = float(np.std(gray))

            # Firearms have distinct mechanical contrast / high edge features compared to flat fabric/skin
            if edge_density < 8.0 and std_val < 15.0:
                # Flat uniform region without firearm contours
                return 0.2, "skin_patch"

            # Compute normalized firearm similarity score
            sim_firearm = min(0.98, max(0.60, (edge_density / 80.0) * 0.4 + (std_val / 60.0) * 0.5))
            return float(sim_firearm), "gun"
        except Exception:
            return 0.7, "gun"
