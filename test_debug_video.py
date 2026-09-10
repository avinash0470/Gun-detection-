import cv2
import sys
from pipeline.config import SystemConfig
from pipeline.core import GunDetectionPipeline

video_path = r"C:\Users\HP\Downloads\WhatsApp Video 2026-09-09 at 3.06.51 PM.mp4"
cap = cv2.VideoCapture(video_path)
cfg = SystemConfig.load_from_file()
pipe = GunDetectionPipeline(cfg)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    guns, _ = pipe.ensemble.detect(frame, 0.0)
    people = pipe.tracker.update(frame, [])
    if guns:
        for g in guns:
            gx1, gy1, gx2, gy2 = g["bbox"]
            h_img, w_img = frame.shape[:2]
            crop = frame[max(0, gy1):min(h_img, gy2), max(0, gx1):min(w_img, gx2)]
            score, cat = pipe.vector_verify.verify_crop(crop)
            res = pipe.process_frame(frame)
            final_dets = len(res.get("detections", []))
            print(f"F{frame_idx}: YOLO={g['confidence']:.2f}, Verif=({score:.2f}, '{cat}'), FinalDets={final_dets}")
cap.release()
print(f"Finished {frame_idx} frames.")
