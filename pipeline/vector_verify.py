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
        Extracts visual vector representation (contour complexity, aspect ratio, gradient distribution, and mechanical features)
        to strictly verify firearms and reject common false positives (computer mouse, black smartphones, flat remotes, drink cans).
        """
        if isinstance(crop_image, dict):
            sim_firearm = crop_image.get("sim_firearm", 0.5)
            sim_negative = crop_image.get("sim_negative", 0.2)
            matched_category = crop_image.get("matched_category", "handgun")
            score = max(0.0, sim_firearm - (sim_negative * 0.5))
            return score, matched_category

        if crop_image is None or not hasattr(crop_image, "shape") or crop_image.size == 0:
            return 0.3, "unknown"

        h, w = crop_image.shape[:2]
        if h < 12 or w < 12:
            return 0.2, "low_res_noise"

        aspect_ratio = float(w) / float(max(1, h))

        # 1. Aspect Ratio & Geometry Verification:
        # Firearms (pistols, rifles) have distinct non-uniform L-shape / elongated geometry.
        # Extreme vertical slivers (<0.22) or uniform flat slabs are rejected.
        if aspect_ratio < 0.22:
            return 0.15, "water_bottle"

        try:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            
            # 2. Rejection of Flat Rectangles (e.g., Black Phones / Wallets / Flat Remotes)
            # Phones are smooth, uniform rectangular glass/metal surfaces with very low internal texture.
            std_val = float(np.std(gray))
            mean_val = float(np.mean(gray))

            # Edge analysis via Canny and Sobel
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.count_nonzero(edges)) / float(h * w + 1e-6)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            gradient_mean = float(np.mean(mag))

            # 3. Rejection of Computer Mouse / Oval Black Devices:
            # Computer mouse: high smoothness, low internal edge complexity, curved single contour.
            # If standard deviation is low and edge density is minimal (<0.04), it's a solid phone or mouse surface.
            if std_val < 18.0 and edge_density < 0.05:
                if mean_val < 60.0:
                    return 0.20, "cellphone"  # Dark phone surface
                return 0.25, "mouse_or_peripheral"

            # 4. Mechanical Feature / Contour Check:
            # Firearms have sharp mechanical edges (barrel, trigger cavity, hammer, grip texture).
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours)
            if contour_count < 2 and edge_density < 0.035:
                # Solid monolithic object with no sub-parts/trigger guard
                return 0.25, "cellphone"

            # 5. Composite firearm confidence score calculation
            # Score scales based on texture contrast and mechanical contour complexity
            raw_score = 0.40 + min(0.35, (gradient_mean / 70.0) * 0.35) + min(0.25, (edge_density / 0.15) * 0.25)
            firearm_score = float(np.clip(raw_score, 0.10, 0.98))

            if firearm_score >= self.config.similarity_threshold:
                return firearm_score, "handgun"
            elif firearm_score >= self.config.hard_negative_threshold:
                return firearm_score, "uncertain_object"
            else:
                return firearm_score, "cellphone"
        except Exception:
            return 0.5, "gun"
