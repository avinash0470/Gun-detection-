import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any

# Try importing PyYAML if installed; fallback to simple YAML parser if missing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

@dataclass
class QualityGateConfig:
    low_light_threshold: float = 0.3
    blur_threshold: float = 100.0
    occlusion_threshold: float = 0.5

@dataclass
class HardwareProfile:
    has_cuda: bool = False
    device_name: str = "CPU"
    device_str: str = "cpu"
    is_high_end_gpu: bool = False
    imgsz: int = 480
    use_half: bool = False
    enable_pose_keypoints: bool = False
    enable_async_reader: bool = True

@dataclass
class DetectorConfig:
    primary_conf_threshold: float = 0.25 # Gun detection confidence threshold
    secondary_conf_threshold: float = 0.20
    person_conf_threshold: float = 0.25  # Person detection confidence threshold
    watch_state_tolerance: float = 0.15
    device: str = "auto"                 # Device option: 'auto', 'cuda', 'cpu', '0', etc.
    imgsz: int = 0                       # 0 means auto-scaled by hardware profile
    half: Optional[bool] = None          # None means auto-decided

@dataclass
class VectorVerifyConfig:
    clip_model_name: str = "ViT-B/32"
    similarity_threshold: float = 0.70
    hard_negative_threshold: float = 0.60

@dataclass
class TemporalConfig:
    history_frames: int = 4
    min_detections_in_window: int = 2
    occlusion_recovery_frames: int = 10
    danger_duration_seconds: float = 3.0

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
class StreamConfig:
    default_source: str = "0"
    rtsp_url: str = "rtsp://admin:hikvision_ipcam@192.168.1.101/Streaming/Channels/101"
    async_capture: bool = True
    buffer_size: int = 1

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
    stream: StreamConfig = field(default_factory=StreamConfig)
    hardware: HardwareProfile = field(default_factory=HardwareProfile)

    @staticmethod
    def detect_hardware(requested_device: str = "auto") -> HardwareProfile:
        """
        Inspects available GPU/CPU resources and selects the optimal scale:
        - High-end GPU (RTX 5000, 4090, 3090, A100, etc.): 640/960px, FP16 half precision, Pose keypoints.
        - Entry/Mid GPU: 640px, FP16 half precision.
        - CPU / No GPU: Adaptive downgrade to 416px, FP32, Fast spatial heuristics.
        """
        profile = HardwareProfile()
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if requested_device.lower() == "cpu":
                has_cuda = False

            if has_cuda:
                profile.has_cuda = True
                dev_idx = 0
                if requested_device.isdigit():
                    dev_idx = int(requested_device)
                profile.device_name = torch.cuda.get_device_name(dev_idx)
                profile.device_str = f"cuda:{dev_idx}" if requested_device.isdigit() else "cuda"
                profile.use_half = True
                
                # Check VRAM or high-end models
                try:
                    vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3)
                except Exception:
                    vram_gb = 8.0

                name_lower = profile.device_name.lower()
                if vram_gb >= 12.0 or "5000" in name_lower or "4090" in name_lower or "3090" in name_lower or "a100" in name_lower or "v100" in name_lower:
                    profile.is_high_end_gpu = True
                    profile.imgsz = 640 # Full scale high-precision for firearm clarity
                    profile.enable_pose_keypoints = True
                else:
                    profile.is_high_end_gpu = False
                    profile.imgsz = 640
                    profile.enable_pose_keypoints = True
            else:
                profile.has_cuda = False
                profile.device_name = "CPU"
                profile.device_str = "cpu"
                profile.is_high_end_gpu = False
                profile.imgsz = 416 # Lightweight scale to sustain 25-30 FPS on CPU without degradation
                profile.use_half = False
                profile.enable_pose_keypoints = False # Fast spatial arm reach heuristics on CPU
        except Exception:
            profile.has_cuda = False
            profile.device_str = "cpu"
            profile.imgsz = 416
            profile.use_half = False
            profile.enable_pose_keypoints = False

        return profile

    @classmethod
    def load_from_file(cls, config_path: str = "config.yaml"):
        """
        Loads user-configurable confidence thresholds from config.yaml or config.json.
        """
        cfg = cls()
        
        # Check candidate configuration files (config.yaml > config.json)
        candidates = [config_path, "config.yaml", "config.json"]
        target_path = None
        for path in candidates:
            if os.path.exists(path):
                target_path = path
                break

        if not target_path:
            return cfg

        try:
            data = {}
            if target_path.endswith(".yaml") or target_path.endswith(".yml"):
                if YAML_AVAILABLE:
                    with open(target_path, "r") as f:
                        data = yaml.safe_load(f) or {}
                else:
                    # Simple fallback parser for basic YAML lines if PyYAML is missing
                    data = cls._parse_simple_yaml(target_path)
            else:
                with open(target_path, "r") as f:
                    data = json.load(f) or {}

            det = data.get("detector", {})
            if "primary_conf_threshold" in det:
                cfg.detector.primary_conf_threshold = float(det["primary_conf_threshold"])
            if "secondary_conf_threshold" in det:
                cfg.detector.secondary_conf_threshold = float(det["secondary_conf_threshold"])
            if "person_conf_threshold" in det:
                cfg.detector.person_conf_threshold = float(det["person_conf_threshold"])
            if "watch_state_tolerance" in det:
                cfg.detector.watch_state_tolerance = float(det["watch_state_tolerance"])
            if "device" in det:
                cfg.detector.device = str(det["device"])

            vec = data.get("vector_verify", {})
            if "similarity_threshold" in vec:
                cfg.vector.similarity_threshold = float(vec["similarity_threshold"])
            if "hard_negative_threshold" in vec:
                cfg.vector.hard_negative_threshold = float(vec["hard_negative_threshold"])

            temp = data.get("temporal", {})
            if "history_frames" in temp:
                cfg.temporal.history_frames = int(temp["history_frames"])
            if "min_detections_in_window" in temp:
                cfg.temporal.min_detections_in_window = int(temp["min_detections_in_window"])
            if "danger_duration_seconds" in temp:
                cfg.temporal.danger_duration_seconds = float(temp["danger_duration_seconds"])

            rsk = data.get("risk", {})
            if "high_threshold" in rsk:
                cfg.risk.high_threshold = float(rsk["high_threshold"])
            if "medium_threshold" in rsk:
                cfg.risk.medium_threshold = float(rsk["medium_threshold"])

            strm = data.get("stream", {})
            if "default_source" in strm:
                cfg.stream.default_source = str(strm["default_source"])
            if "rtsp_url" in strm:
                cfg.stream.rtsp_url = str(strm["rtsp_url"])
            if "async_capture" in strm:
                cfg.stream.async_capture = bool(strm["async_capture"])

        except Exception as e:
            print(f"Warning: Could not parse configuration file '{target_path}': {e}")

        # Compute hardware profile
        cfg.hardware = cls.detect_hardware(cfg.detector.device)
        return cfg

    @staticmethod
    def _parse_simple_yaml(filepath: str) -> Dict:
        """Fallback lightweight parser for basic flat/nested key-value YAML files."""
        result = {}
        current_section = None
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if not val:
                        current_section = key
                        result[current_section] = {}
                    else:
                        # strip inline comments
                        if "#" in val:
                            val = val.split("#")[0].strip()
                        try:
                            parsed_val = float(val) if "." in val else int(val)
                        except ValueError:
                            parsed_val = val
                        if current_section:
                            result[current_section][key] = parsed_val
                        else:
                            result[key] = parsed_val
        return result
