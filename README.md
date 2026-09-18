# Multi-Stage Gun Detection & Verification System

An enterprise-grade, multi-stage real-time firearm detection and verification framework designed for camera arrays (Webcam / RTSP CCTV feeds) and recorded video analytics.

## Pipeline Architecture

```text
CAMERA (Webcam / RTSP Stream / Video File)
   │
   ▼
FRAME PROCESSING & QUALITY GATE ── low light / blur / occlusion score → adjusts downstream thresholds
   │
   ▼
PERSON DETECTION + PERSISTENT TRACKING (YOLO11m + ByteTrack / ReID appearance vectors)
   │
   ▼
FIREARM DETECTION (YOLO: best_gun_26n(23).pt)
   │        → Candidate Bounding Box + Confidence score
   ▼
CROP + CONTEXTUAL RESIZE (20% spatial padding margin to preserve hand grip context)
   │
   ▼
DEEP LEARNING CROP CLASSIFIER (MobileNetV3)
   │        → Gun Confirmation (Revolver / Rifle / Assault Rifle / Holster)
   │        → Hard Negative Rejection (Cellphones, Drink Bottles, Computer Mice, Tools, Remotes)
   ▼
PERSON-GUN ASSOCIATION (Spatial reach & wrist proximity check)
   │
   ▼
MULTI-FRAME TEMPORAL VALIDATION (Persistent tracking across sliding window N frames)
   │
   ▼
THREAT RISK SCORING (Detection conf + pose context + location risk + track stability + vector score)
   │
   ▼
FINAL ALERT & VISUAL OVERLAYS
   ├── Clean individual → Green Box (ID hidden)
   └── ARMED SUSPECT (DANGER) → Bold Red Box (SHA-256 evidence hashing + snapshot logging)
```

---

## Hardware Auto-Profiling & Dynamic Scaling

The pipeline automatically inspects your system hardware upon startup and adjusts the entire pipeline scale without requiring manual re-configuration:

- **NVIDIA High-End GPU (RTX 5000, 4090, 3090, A100, etc.)**:
  - **Full Scale Mode**: `imgsz=640` (or `960`), FP16 Half-Precision (`half=True`), full YOLO pose estimation keypoint verification.
  - Zero lag high-FPS processing for maximum weapon detail.
- **CPU / No GPU (Laptop / Low-resource systems)**:
  - **Adaptive Downgrade Mode**: `imgsz=416`, FP32 precision, lightweight spatial arm-reach heuristics (skips heavy pose model to prevent CPU throttling & stutter).
- **Adaptive Frame Ingestion**:
  - **Live Streams (RTSP / Webcam)**: Multithreaded bufferless reader (`CAP_PROP_BUFFERSIZE = 1`) with TCP transport to eliminate video lag and packet corruption.
  - **Recorded Videos (`.mp4`, `.avi`)**: Sequential frame-by-frame processing ensuring **zero frame skipping** so every single frame is thoroughly inspected.

> [!TIP]
> If you have an NVIDIA GPU (e.g., RTX 5000) but PyTorch detects CPU, ensure CUDA-enabled PyTorch is installed:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Configuration (`config.yaml`) & Setup

### 1. Create and Activate Virtual Environment

Open PowerShell in the project directory:

```powershell
# Create virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

*Note: On Command Prompt (cmd.exe), activate using:*
```cmd
venv\Scripts\activate.bat
```

### 2. Install Dependencies

#### For CPU Only:
```powershell
pip install -r requirements.txt
```

#### For NVIDIA GPU (CUDA Acceleration) 🚀:
To run models on GPU (RTX / GTX cards) for high FPS (30–60+ FPS):
```powershell
# 1. Install PyTorch with CUDA support (e.g. CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies
pip install -r requirements.txt
```

Verify GPU availability:
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Usage Guide

### Option A: Run Live Webcam Feed (`--source 0`)

```powershell
python main.py --source 0
```

### Option B: Run Live RTSP CCTV Camera Feed

Configure your camera stream URL in `config.yaml` (`stream.rtsp_url`):
```yaml
stream:
  rtsp_url: "rtsp://<username>:<password>@<camera-ip>/Streaming/Channels/101"
```
Then run using the `--rtsp` flag:
```powershell
python main.py --rtsp
```
Or specify a custom RTSP stream directly on the command line:
```powershell
python main.py --source "rtsp://<username>:<password>@<camera-ip>/Streaming/Channels/101"
```

### Option C: Run on Video File and Save Output

```powershell
python main.py --source "path/to/video.mp4" --save --output-dir "output"
```

### Option D: Explicit GPU / CPU Device Selection

The pipeline automatically selects GPU if available. You can also explicitly specify the target device:
```powershell
# Force CUDA GPU 0
python main.py --source 0 --device 0

# Force CPU
python main.py --source 0 --device cpu
```

### Option E: Configure Location Threat Risk (`--location-risk`)

Configure baseline location threat risk (between `0.0` for low risk like a gun range and `1.0` for high risk like a school lobby/bank):
```powershell
python main.py --source 0 --location-risk 0.8
```
Or set `stream.location_risk: 0.8` globally in `config.yaml`.

### Option F: Run Pipeline Simulation Test Harness

```powershell
python main.py
```

Press `q` to quit the live feed.

---

## Output & Evidence

### Video Recordings

All live feed sessions are **automatically recorded** to the `recordings/` directory. Files are saved as timestamped `.avi` files (e.g. `detection_2026-09-03_14-08-15.avi`).

### Evidence Snapshots

When a **DANGER** alert is triggered, the system automatically saves:
- A **JPEG snapshot** of the frame to the `evidence/` directory
- A **SHA-256 evidence hash** for chain-of-custody verification

### Detection Event Logs

All detection events (LOW, MEDIUM, HIGH, DANGER) are logged to `logs/detection_events.jsonl` in JSON Lines format for audit trail and post-incident analysis.

---

## Visual Overlay Guide

- **Thin Green Box**: Clean / Unarmed individual (ID label hidden).
- **Yellow/Orange Box (`SUSPECT` / `SUSPECT (WEAPON CONCEALED)`)**: Medium risk candidate or person who has hidden a previously visible weapon.
- **Bold Red Box (`DANGER! ARMED SUSPECT`)**: High risk threat holding a firearm for longer than 3 seconds. Triggers cryptographic evidence hashing.
