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

## Installation & Setup

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

With the virtual environment activated:

```powershell
pip install -r requirements.txt
```

---

## Usage Guide

### Option A: Run Live Webcam Feed (`--source 0`)

```powershell
python main.py --source 0
```

### Option B: Run Live RTSP CCTV Camera Feed

```powershell
python main.py --source "rtsp://username:password@ip_address:port/h264"
```

### Option C: Run Pipeline Simulation Test Harness

```powershell
python main.py
```

---

## Visual Overlay Guide

- **Thin Green Box**: Clean / Unarmed individual (ID label hidden).
- **Yellow/Orange Box (`SUSPECT` / `SUSPECT (WEAPON CONCEALED)`)**: Medium risk candidate or person who has hidden a previously visible weapon.
- **Bold Red Box (`DANGER! ARMED SUSPECT`)**: High risk threat holding a firearm for longer than 3 seconds. Triggers cryptographic evidence hashing.
