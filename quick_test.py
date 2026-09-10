import cv2
from pipeline.config import SystemConfig
from pipeline.core import GunDetectionPipeline

video_path = r"C:\Users\HP\Downloads\WhatsApp Video 2026-09-09 at 3.06.51 PM.mp4"
cap = cv2.VideoCapture(video_path)
cfg = SystemConfig.load_from_file()
pipe = GunDetectionPipeline(cfg)

f_idx = 0
found = 0
while True:
    ret, frame = cap.read()
    if not ret or found >= 5:
        break
    f_idx += 1
    guns, _ = pipe.ensemble.detect(frame, 0.0)
    if guns:
        found += 1
        g = guns[0]
        gx1, gy1, gx2, gy2 = g['bbox']
        h_img, w_img = frame.shape[:2]
        crop = frame[max(0, gy1):min(h_img, gy2), max(0, gx1):min(w_img, gx2)]
        score, cat = pipe.vector_verify.verify_crop(crop)
        print(f"Frame {f_idx}: Box={g['bbox']} Conf={g['confidence']:.2f} -> Score={score:.2f}, Category='{cat}'")
cap.release()
