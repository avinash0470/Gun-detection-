import cv2
import argparse
import logging
import json
import time
from pipeline.config import SystemConfig
from pipeline.core import GunDetectionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GunDetectionPipeline")

def run_simulation(pipeline):
    # Test Frame 1: Person enters holding an aimed handgun -> DANGER
    frame_1 = {
        "brightness": 0.7,
        "blur": 120.0,
        "occlusion": 0.0,
        "people": [{"bbox": (100, 100, 200, 400)}],
        "detections_primary": [{"bbox": (110, 180, 140, 210), "confidence": 0.92}],
        "detections_secondary": [{"bbox": (108, 178, 142, 212), "confidence": 0.88}],
        "crop_sim_info": {"sim_firearm": 0.95, "sim_negative": 0.05, "matched_category": "handgun"},
        "pose_context": "aimed",
        "sim_duration_sec": 4.2
    }

    # Test Frame 2: Weapon concealed / hidden (0 firearm detections!)
    # Pipeline MUST hold person in SUSPECT (WEAPON CONCEALED) state
    frame_2 = {
        "brightness": 0.7,
        "blur": 120.0,
        "occlusion": 0.0,
        "people": [{"bbox": (100, 100, 200, 400)}],
        "detections_primary": [],
        "detections_secondary": [],
        "pose_context": "casual"
    }

    print("=== RUNNING CONCEALMENT & STICKY SUSPECT TEST ===")
    out_1 = pipeline.process_frame(frame_1, location_risk=0.8)
    print("Frame 1 (Active Gun Armed Threat):", json.dumps(out_1, indent=2))
    
    # Process frame 2 where gun is hidden
    out_2 = pipeline.process_frame(frame_2, location_risk=0.8)
    print("\nFrame 2 (Weapon Concealed / Hidden - 0 Gun Detections):", json.dumps(out_2, indent=2))

def run_live_feed(pipeline, source_input):
    try:
        source = int(source_input)
    except ValueError:
        source = source_input

    logger.info(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video source {source}.")
        return

    cv2.namedWindow("Gun Detection Pipeline", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame.")
            break

        output = pipeline.process_frame(frame, location_risk=0.5)
        persons = output.get("persons", [])
        detections = output.get("detections", [])
        quality = output.get("quality", {})

        threatened_track_ids = {}
        for det in detections:
            tid = det.get("track_id")
            if tid and not tid.startswith("gun_unassociated"):
                threatened_track_ids[tid] = det

        # 1. Render Person Bounding Boxes
        for person in persons:
            px1, py1, px2, py2 = person["bbox"]
            pid = person["track_id"]
            is_suspect = person.get("is_suspect", False)
            is_concealed = person.get("is_concealed", False)

            if pid in threatened_track_ids:
                det_info = threatened_track_ids[pid]
                risk_level = det_info["risk_level"]
                duration = det_info.get("gun_duration_sec", 0.0)
                
                if risk_level == "DANGER" or risk_level == "HIGH":
                    p_color = (0, 0, 255) # Red
                    label = f"DANGER! ARMED SUSPECT ({duration:.1f}s)"
                    cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 4)
                    cv2.putText(frame, label, (px1, max(py1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, p_color, 2)
                else:
                    p_color = (0, 215, 255) # Yellow/Orange
                    label = "SUSPECT"
                    cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 2)
                    cv2.putText(frame, label, (px1, max(py1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2)
            elif is_concealed or is_suspect:
                # WEAPON CONCEALED / HIDDEN STATE (Holds Yellow/Orange warning box)
                p_color = (0, 215, 255) # Yellow/Orange
                label = "SUSPECT (WEAPON CONCEALED)"
                cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 2)
                cv2.putText(frame, label, (px1, max(py1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2)
            else:
                # Clean person (thin green box, no text label)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 1)

        # 2. Render Firearm Detections
        for det in detections:
            gx1, gy1, gx2, gy2 = det["gun_bbox"]
            risk_level = det["risk_level"]
            risk_score = det["risk_score"]
            conf = det["confidence"]
            cls_name = det["class"]
            duration = det.get("gun_duration_sec", 0.0)

            if risk_level == "DANGER" or risk_level == "HIGH":
                color = (0, 0, 255)
                status_tag = f"DANGER ALERT ({duration:.1f}s)"
            elif risk_level == "MEDIUM":
                color = (0, 215, 255)
                status_tag = "SUSPECTED"
            else:
                color = (0, 255, 0)
                status_tag = "LOW RISK"

            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, 3)
            
            label = f"{cls_name.upper()} ({conf:.2f}) | {status_tag}"
            cv2.putText(frame, label, (gx1, max(gy1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Header status bar overlay
        status_txt = f"Quality Offset: +{quality.get('threshold_offset', 0):.2f} | Active Threat Alerts: {len(detections)}"
        cv2.putText(frame, status_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Gun Detection Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Video capture released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun Detection and Verification Pipeline")
    parser.add_argument("--source", type=str, default=None, 
                        help="Video source index (e.g. '0' for Webcam) or RTSP Stream URL. If omitted, runs simulated test.")
    args = parser.parse_args()

    config = SystemConfig()
    pipeline = GunDetectionPipeline(config)

    if args.source is not None:
        run_live_feed(pipeline, args.source)
    else:
        run_simulation(pipeline)
