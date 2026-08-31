from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class QualityGateConfig:
    low_light_threshold: float = 0.3    # Below this is low light
    blur_threshold: float = 100.0       # Laplacian variance threshold
    occlusion_threshold: float = 0.5    # Mock occlusion score threshold

@dataclass
class DetectorConfig:
    primary_conf_threshold: float = 0.45 # Increased from 0.25 to filter out everyday objects like bottles/phones
    secondary_conf_threshold: float = 0.40
    watch_state_tolerance: float = 0.15

@dataclass
class VectorVerifyConfig:
    clip_model_name: str = "ViT-B/32"
    similarity_threshold: float = 0.75   # Raised cutoff to eliminate false positives
    hard_negative_threshold: float = 0.60

@dataclass
class TemporalConfig:
    history_frames: int = 5
    min_detections_in_window: int = 3
    occlusion_recovery_frames: int = 10

@dataclass
class RiskConfig:
    high_threshold: float = 0.75
    medium_threshold: float = 0.45
    weights: Dict[str, float] = field(default_factory=lambda: {
        "detection_conf": 0.35,
        "pose_risk": 0.25,
        "location_risk": 0.10,
        "track_stability": 0.10,
        "vector_score": 0.20
    })

@dataclass
class SystemConfig:
    degraded_mode: bool = False
    enable_ensemble: bool = True
    enable_vector_db: bool = True
    quality: QualityGateConfig = field(default_factory=QualityGateConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    vector: VectorVerifyConfig = field(default_factory=VectorVerifyConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
