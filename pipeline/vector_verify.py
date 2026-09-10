import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from .config import VectorVerifyConfig

logger = logging.getLogger("GunDetectionPipeline")

try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class VectorVerification:
    """
    Two-Tiered Crop Classifier & Verification Engine:
    1. Deep Learning Classifier (MobileNetV3-Small):
       Infers semantic visual concepts on cropped regions. Distinguishes firearms (revolver, rifle, assault rifle, holster)
       from common false positive distractors (cellular phone, wallet, remote control, power drill, water bottle, mouse).
    2. Structural & Edge Fallback:
       Applies contour & gradient analysis when neural inference is unavailable or in degraded mode.
    """
    def __init__(self, config: VectorVerifyConfig, hardware_profile=None):
        self.config = config
        self.hardware = hardware_profile
        self.device = "cpu"
        if self.hardware and getattr(self.hardware, "device_str", None):
            self.device = self.hardware.device_str

        # Categories considered confirmed firearms
        self.firearm_categories = {"revolver", "rifle", "assault rifle", "holster", "handgun"}
        
        # Hard negative distractors that commonly cause false alarms in CCTV
        self.hard_negatives = [
            "cellphone", "cellular telephone", "mobile_phone", "water_bottle", 
            "pop bottle", "beer bottle", "drink_can", "power drill", "screwdriver", 
            "wallet", "remote control", "remote_control", "mouse", "toy_gun", 
            "skin_patch", "low_res_noise"
        ]

        # Neural classifier initialization
        self.model = None
        self.transform = None
        self.categories = None
        self.firearm_indices = []
        self.negative_indices = []

        if TORCH_AVAILABLE and getattr(self.config, "enable_dl_classifier", True):
            self._init_classifier()

    def _init_classifier(self):
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
            self.model.eval()
            if self.device != "cpu":
                try:
                    self.model.to(self.device)
                except Exception:
                    self.device = "cpu"
                    self.model.to("cpu")

            self.transform = weights.transforms()
            self.categories = weights.meta["categories"]

            # Pre-index firearm classes in ImageNet
            for idx, name in enumerate(self.categories):
                name_l = name.lower()
                if any(w in name_l for w in ["revolver", "rifle", "assault rifle", "holster"]):
                    if "trifle" not in name_l:
                        self.firearm_indices.append(idx)
                elif any(w in name_l for w in ["cellular telephone", "remote control", "power drill", 
                                               "wallet", "water bottle", "beer bottle", "mouse"]):
                    self.negative_indices.append(idx)

            logger.info(f"Loaded Deep Learning Crop Classifier (MobileNetV3) on device={self.device}.")
        except Exception as e:
            logger.warning(f"Could not initialize deep learning crop classifier: {e}. Falling back to structural verification.")
            self.model = None

    def verify_crop(self, crop_image) -> Tuple[float, str]:
        """
        Classifies cropped region:
        Returns:
            Tuple[float, str]: (confidence_score, predicted_category)
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
        if h < 14 or w < 14:
            return 0.2, "low_res_noise"

        # Tier 1: Deep Learning Classification (MobileNetV3)
        if self.model is not None and TORCH_AVAILABLE:
            try:
                # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
                rgb_crop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_img = Image.fromarray(rgb_crop)
                
                input_tensor = self.transform(pil_img).unsqueeze(0)
                if self.device != "cpu":
                    input_tensor = input_tensor.to(self.device)

                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0]

                # Aggregate probabilities for firearm classes vs hard negative distractor classes
                firearm_prob = float(torch.sum(probs[self.firearm_indices])) if self.firearm_indices else 0.0
                negative_prob = float(torch.sum(probs[self.negative_indices])) if self.negative_indices else 0.0
                top_class_idx = int(torch.argmax(probs))
                top_class_name = self.categories[top_class_idx]

                # 1. If top predicted class is explicitly a known non-firearm / distractor
                for neg in self.hard_negatives:
                    if neg in top_class_name.lower():
                        return 0.10, top_class_name

                # 2. Strict DL Verification:
                # If the neural classifier sees virtually zero firearm probability (< 0.05) or
                # the top predicted class is NOT in firearm categories (e.g. clothing, armor, bottle, phone, wheel)
                is_top_firearm = any(w in top_class_name.lower() for w in ["revolver", "rifle", "assault rifle", "holster"])
                
                if not is_top_firearm and firearm_prob < 0.05:
                    # Reject object as non-gun
                    return 0.15, top_class_name

                # 3. If firearm probability is confirmed
                if is_top_firearm or firearm_prob >= 0.05:
                    final_score = float(np.clip(0.60 + firearm_prob * 0.40 - (negative_prob * 0.20), 0.50, 0.99))
                    return final_score, "handgun"
                else:
                    return 0.20, top_class_name
            except Exception as e:
                logger.debug(f"Classifier inference exception: {e}")

        # Tier 2: Structural / Geometric fallback (only reached if model is None)
        return self._structural_verification(crop_image)

    def _structural_verification(self, crop_image) -> Tuple[float, str]:
        """Fast fallback geometric & gradient verification."""
        h, w = crop_image.shape[:2]
        aspect_ratio = float(w) / float(max(1, h))

        if aspect_ratio < 0.20:
            return 0.15, "water_bottle"

        try:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            std_val = float(np.std(gray))
            mean_val = float(np.mean(gray))

            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.count_nonzero(edges)) / float(h * w + 1e-6)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            gradient_mean = float(np.mean(mag))

            # Flat uniform dark rectangle (smartphone screen/back)
            if std_val < 18.0 and edge_density < 0.05:
                if mean_val < 65.0:
                    return 0.20, "cellphone"
                return 0.25, "mouse"

            raw_score = 0.40 + min(0.35, (gradient_mean / 70.0) * 0.35) + min(0.25, (edge_density / 0.15) * 0.25)
            firearm_score = float(np.clip(raw_score, 0.10, 0.98))

            if firearm_score >= self.config.similarity_threshold:
                return firearm_score, "handgun"
            elif firearm_score >= self.config.hard_negative_threshold:
                return firearm_score, "uncertain_object"
            else:
                return firearm_score, "cellphone"
        except Exception:
            return 0.5, "handgun"
