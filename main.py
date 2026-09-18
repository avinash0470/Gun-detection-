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

import threading

class ThreadedVideoCapture:
    """
    Dedicated background frame reader that continuously grabs the freshest frame
    from RTSP / Webcam. Prevents OpenCV internal buffer build-up, latency lag, and network degradation.
    """
    def __init__(self, source):
        self.source = source
        self.running = True
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()

        # Configure FFMPEG options for low latency and TCP stability
        if isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("rtsps://")):
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def _capture_loop(self):
        while self.running:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
                else:
                    self.running = False
                    break
            time.sleep(0.005) # Tiny sleep to prevent 100% core spin

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def isOpened(self):
        return self.cap.isOpened() and self.running

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

def run_live_feed(pipeline, source_input, save_video=False, output_dir="output", location_risk=None):
    import os
    if location_risk is None:
        location_risk = getattr(pipeline.config.stream, "location_risk", 0.0)
    try:
        source = int(source_input)
    except ValueError:
        source = source_input

    logger.info(f"Opening video source: {source}")
    
    # Use Threaded Capture ONLY for real-time live sources (RTSP or Webcam)
    # For local recorded video files (.mp4, .avi, etc.), read sequentially so NO frames are skipped
    is_live_stream = isinstance(source, int) or (isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("rtsps://") or source.isdigit()))
    use_async = getattr(pipeline.config.stream, "async_capture", True) and is_live_stream

    if use_async:
        logger.info("Using ThreadedVideoCapture (bufferless real-time mode for live stream).")
        cap = ThreadedVideoCapture(source)
        time.sleep(0.3)
    else:
        if not is_live_stream:
            logger.info("Processing recorded video file sequentially (every frame analyzed, zero frame skipping).")
            if isinstance(source, str) and not os.path.exists(source):
                logger.error(f"Error: Video file not found: '{source}'. Please verify the file path.")
                return
        if isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("rtsps://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video source {source}.")
        return

    # Configure Full Screen Window
    window_name = "Gun Detection Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    # Video Writer setup if saving is requested
    writer = None
    if save_video:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output filename
        if isinstance(source_input, str) and os.path.exists(source_input):
            base_name = os.path.splitext(os.path.basename(source_input))[0]
            out_filename = f"detected_{base_name}.mp4"
        else:
            out_filename = f"detected_output_{int(time.time())}.mp4"
            
        output_path = os.path.join(output_dir, out_filename)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        save_fps = src_fps if src_fps and src_fps > 0 else 25.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, save_fps, (frame_width, frame_height))
        logger.info(f"Saving output video to: {output_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame or reached end of video.")
                break

            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # Execute pipeline process_frame
            output = pipeline.process_frame(frame, location_risk=location_risk)
            persons = output.get("persons", [])
            detections = output.get("detections", [])
            quality = output.get("quality", {})

            # Map detections to associated person IDs
            threatened_track_ids = {}
            for det in detections:
                tid = det.get("track_id")
                if tid and not tid.startswith("gun_unassociated") and det.get("is_in_hand", False):
                    threatened_track_ids[tid] = det

            # =========================================================================
            # STAGE 8 & VISUAL OVERLAYS:
            # 1. Thin Green Box: Clean / Unarmed individual (ID label HIDDEN)
            # 2. Bold Red/Orange Box (ARMED SUSPECT): Only shown when an active gun is in hand
            # =========================================================================
            for person in persons:
                px1, py1, px2, py2 = person["bbox"]
                pid = person["track_id"]

                # Check if this person actively holds a verified firearm in hand
                if pid in threatened_track_ids:
                    det_info = threatened_track_ids[pid]
                    duration = det_info.get("gun_duration_sec", 0.0)
                    risk_lvl = det_info.get("risk_level", "MEDIUM")
                    
                    # Red ONLY if verified DANGER (high confidence + high vector match + >3s duration)
                    p_color = (0, 0, 255) if risk_lvl == "DANGER" else (0, 140, 255) # Red / Orange
                    label = f"ARMED SUSPECT [{risk_lvl}] [ID: {pid}]"
                    if duration > 1.0:
                        label += f" ({duration:.1f}s)"

                    cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 2)
                    cv2.putText(frame, label, (px1, max(py1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)
                elif person.get("is_concealed", False):
                    # Person was previously armed; weapon is currently concealed/hidden
                    p_color = (0, 165, 255) # Orange
                    label = f"SUSPECT (WEAPON CONCEALED) [ID: {pid}]"
                    cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 2)
                    cv2.putText(frame, label, (px1, max(py1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)
                else:
                    # Clean / Unarmed individual: Green box with NO ID / NO text
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 1)

            # Render Firearm Bounding Boxes (Red)
            for det in detections:
                gx1, gy1, gx2, gy2 = det["gun_bbox"]
                conf = det["confidence"]
                tid = det.get("track_id", "")
                is_unattended = tid.startswith("gun_unassociated") or not det.get("is_in_hand", False)
                cls_name = "GUN (UNATTENDED)" if is_unattended else "GUN"
                duration = det.get("gun_duration_sec", 0.0)

                color = (0, 0, 255) # Red
                label = f"{cls_name} ({conf:.2f})"
                if duration > 1.0:
                    label += f" [{duration:.1f}s]"

                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, 2)
                cv2.putText(frame, label, (gx1, max(gy1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Header status bar overlay
            threat_count = len(detections)
            gun_status = 1 if threat_count > 0 else 0
            dev_tag = f"{pipeline.hardware.device_name} ({pipeline.hardware.imgsz}px)"
            
            # Real-time terminal printout: Gun: 1 (armed person ID) or Gun: 0
            if gun_status == 1:
                armed_ids = [d["track_id"] for d in detections if d.get("track_id") and not d.get("track_id", "").startswith("gun_unassociated")]
                if armed_ids:
                    print(f"[STATUS] GUN: 1 | Armed Person ID: {', '.join(armed_ids)} | Detections: {threat_count} | FPS: {current_fps:.1f} | Dev: {dev_tag}", flush=True)
                else:
                    print(f"[STATUS] GUN: 1 (UNATTENDED) | Detections: {threat_count} | FPS: {current_fps:.1f} | Dev: {dev_tag}", flush=True)
            else:
                print(f"[STATUS] GUN: 0 | People: {len(persons)} | FPS: {current_fps:.1f} | Dev: {dev_tag}", flush=True)

            status_txt = f"FPS: {current_fps:.1f} | {dev_tag} | People: {len(persons)} | GUN: {gun_status}"
            cv2.putText(frame, status_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if gun_status == 1 else (0, 255, 0), 2)

            # Write annotated frame to output video if writer is initialized
            if writer is not None:
                writer.write(frame)

            cv2.imshow("Gun Detection Pipeline", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info("Output video successfully saved and closed.")
        cv2.destroyAllWindows()
        logger.info("Video capture released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun Detection and Verification Pipeline")
    parser.add_argument("--source", type=str, default=None, 
                        help="Video source: '0' (Webcam), 'rtsp' (uses config.yaml RTSP URL), RTSP URL string, or 'sim' for simulation mode.")
    parser.add_argument("--rtsp", action="store_true", help="Shortcut flag to run configured RTSP stream from config.yaml")
    parser.add_argument("--save", action="store_true", help="Save the annotated detection video to an output directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory where the processed video will be saved (default: 'output')")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cuda', 'cpu', '0', '1', or 'auto' (default: auto-detected)")
    parser.add_argument("--location-risk", type=float, default=None, help="Location baseline risk score between 0.0 (low risk, e.g. gun range) and 1.0 (high risk, e.g. school lobby). Defaults to config.yaml setting or 0.0.")
    parser.add_argument("--gun-model", type=str, default=None, help="Path to gun detection YOLO model weights (.pt). Defaults to config.yaml setting or auto-selected.")
    parser.add_argument("--require-person", dest="require_person", action="store_true", default=None, help="Require gun to be held by a person.")
    parser.add_argument("--detect-without-person", "--no-require-person", dest="require_person", action="store_false", default=None, help="Detect guns anywhere, even without a person.")
    args = parser.parse_args()

    # Load configuration from config.yaml
    config = SystemConfig.load_from_file("config.yaml")
    if args.gun_model is not None:
        config.detector.gun_model = args.gun_model
    if args.device is not None:
        config.detector.device = args.device
    if args.location_risk is not None:
        config.stream.location_risk = args.location_risk
    if args.require_person is not None:
        config.detector.require_person = args.require_person

    pipeline = GunDetectionPipeline(config)

    # Resolve video source input
    source = args.source
    if args.rtsp or (isinstance(source, str) and source.lower() == "rtsp"):
        source = config.stream.rtsp_url
    elif source is None:
        def_src = config.stream.default_source
        if isinstance(def_src, str) and def_src.lower() == "rtsp":
            source = config.stream.rtsp_url
        elif def_src and str(def_src).lower() != "sim":
            source = def_src

    if source is not None and str(source).lower() != "sim":
        run_live_feed(pipeline, source, save_video=args.save, output_dir=args.output_dir, location_risk=config.stream.location_risk)
    else:
        run_simulation(pipeline)
