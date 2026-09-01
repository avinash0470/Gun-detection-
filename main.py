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

    # Flushes buffer to eliminate camera lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Gun Detection Pipeline", cv2.WINDOW_NORMAL)

    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame.")
            break

        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()

        output = pipeline.process_frame(frame, location_risk=0.5)
        persons = output.get("persons", [])
        detections = output.get("detections", [])
        quality = output.get("quality", {})

        threatened_track_ids = {}
        for det in detections:
            tid = det.get("track_id")
            if tid and not tid.startswith("gun_unassociated"):
                threatened_track_ids[tid] = det

        # 1. Render Person Bounding Boxes & Formatted ReID Tags
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
                    label = f"DANGER! ARMED SUSPECT [ReID: {pid}] ({duration:.1f}s)"
                    cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 4)
                    cv2.putText(frame, label, (px1, max(py1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, p_color, 2)
                else:
                    p_color = (0, 215, 255) # Yellow/Orange
                    label = f"SUSPECT [ReID: {pid}]"
                    cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 2)
                    cv2.putText(frame, label, (px1, max(py1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2)
            elif is_concealed or is_suspect:
                # WEAPON CONCEALED / HIDDEN STATE (Display ReID tag for suspect)
                p_color = (0, 215, 255) # Yellow/Orange
                label = f"SUSPECT (CONCEALED) [ReID: {pid}]"
                cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 2)
                cv2.putText(frame, label, (px1, max(py1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2)
            else:
                # Clean person (thin green box, NO text label)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 1)

        # 2. Render Firearm Bounding Boxes
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

        # Header status bar overlay with real FPS counter
        status_txt = f"FPS: {current_fps:.1f} | Quality Offset: +{quality.get('threshold_offset', 0):.2f} | Threat Alerts: {len(detections)}"
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

    # Load configuration from config.yaml
    config = SystemConfig.load_from_file("config.yaml")
    pipeline = GunDetectionPipeline(config)

    if args.source is not None:
        run_live_feed(pipeline, args.source)
    else:
        run_simulation(pipeline)
