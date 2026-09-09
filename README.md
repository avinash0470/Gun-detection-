# Multi-Stage Gun Detection & Verification System

An enterprise-grade, multi-stage real-time firearm detection and verification framework designed for camera arrays (Webcam / RTSP CCTV feeds).

## Pipeline Architecture

```text
CAMERA (webcam / RTSP stream)
   │
   ▼
FRAME QUALITY GATE ── low light / blur / occlusion score → adjusts downstream thresholds
   │
   ▼
PERSON DETECTION + TRACKING (yolo11m + ReID vector gallery for re-entry/occlusion recovery)
   │
   ▼
PARALLEL DETECTION ENSEMBLE (Primary: best_gun_11m.pt | Secondary)
   │        → disagreement = "watch state" (prevents premature rejection)
   ▼
Candidate Bounding Box + confidence
   │
   ▼
Crop Object → CLIP Vector Verification (Firearm vs Hard-Negatives like bottles, phones, tools)
   │
   ▼
MULTI-FRAME TEMPORAL VALIDATION (persistent tracking across N frames, exposure duration timer)
   │
   ▼
POSE / CONTEXT CHECK (aimed, holstered, carried, concealed)
   │
   ▼
PERSON-GUN ASSOCIATION (ReID-linked tracking)
   │
   ▼
RISK SCORING (detection conf + pose + location risk + track stability + vector score)
   │
   ├── Low → Keep watching, log only
   ├── Medium → SUSPECT state / Queue for human review
   └── High/Danger → ARMED THREAT / DANGER ALERT (SHA-256 evidence hashing + instant alert)
   │
   ▼
STICKY SUSPECT MEMORY (If weapon is concealed/hidden, suspect status is retained as "SUSPECT (WEAPON CONCEALED)")
   │
   ▼
HUMAN REVIEW & FEEDBACK LOOP (Operator clearance API protocol)
```

---

## Hardware Auto-Profiling & Dynamic Scaling

The pipeline automatically inspects your system hardware upon startup and adjusts the entire pipeline scale without requiring manual re-configuration:

- **NVIDIA High-End GPU (RTX 5000, 4090, 3090, A100, etc.)**:
  - **Full Scale Mode**: `imgsz=640` (or `960`), FP16 Half-Precision (`half=True`), full YOLO pose estimation keypoint verification.
  - Zero lag high-FPS processing for maximum weapon detail.
- **CPU / No GPU (Laptop / Low-resource systems)**:
  - **Adaptive Downgrade Mode**: `imgsz=416`, FP32 precision, lightweight spatial arm-reach heuristics (skips heavy pose model to prevent CPU throttling & stutter).
- **Asynchronous Threaded Frame Ingestion**:
  - Automatically captures RTSP & Webcam frames in a background worker thread with `CAP_PROP_BUFFERSIZE = 1` and TCP transport to eliminate video lag, packet-loss corruption, and frame drops.

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

```powershell
python main.py --source "rtsp://admin:password@192.168.1.101/Streaming/Channels/102"
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

### Option E: Run Pipeline Simulation Test Harness

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
