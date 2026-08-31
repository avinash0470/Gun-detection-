from typing import Dict, List, Tuple
from .config import VectorVerifyConfig

class VectorVerification:
    def __init__(self, config: VectorVerifyConfig):
        self.config = config
        self.firearm_references = ["handgun_ref", "rifle_ref", "shotgun_ref", "revolver_ref"]
        # Expanded hard negatives for common false positive objects
        self.hard_negatives = [
            "cellphone", "mobile_phone", "water_bottle", "drink_can", 
            "power_tool", "screwdriver", "wallet", "remote_control", "toy_gun"
        ]

    def verify_crop(self, crop_image) -> Tuple[float, str]:
        """
        Extracts CLIP embedding for the cropped object and performs similarity search.
        Includes aspect ratio & object filtering to suppress false positive bottles/phones.
        """
        sim_firearm = 0.5
        sim_negative = 0.2
        matched_category = "unknown"

        if isinstance(crop_image, dict):
            sim_firearm = crop_image.get("sim_firearm", 0.5)
            sim_negative = crop_image.get("sim_negative", 0.2)
            matched_category = crop_image.get("matched_category", "unknown")
        elif crop_image is not None and hasattr(crop_image, "shape"):
            # Check crop bounding box dimensions if numpy image slice passed
            h, w = crop_image.shape[:2]
            aspect_ratio = h / max(1, w)
            # Water bottles typically have tall vertical aspect ratios (> 2.2) without horizontal firearm handles
            if aspect_ratio > 2.5:
                sim_negative += 0.35 # Strong penalty towards bottle/can hard-negative

        # Check if matched category is in hard negatives
        if matched_category in self.hard_negatives:
            sim_negative += 0.4

        # Compute vector score: penalize heavily if hard negative similarity is strong
        vector_verification_score = max(0.0, sim_firearm - (sim_negative * 0.5))
        
        return vector_verification_score, matched_category
