import cv2
import torch
from pipeline.config import SystemConfig
from pipeline.core import GunDetectionPipeline

video_path = r"C:\Users\HP\Downloads\WhatsApp Video 2026-09-09 at 3.06.51 PM.mp4"
cap = cv2.VideoCapture(video_path)
cfg = SystemConfig.load_from_file()
pipe = GunDetectionPipeline(cfg)

ret, frame = cap.read()
ret, frame = cap.read() # frame 2
guns, _ = pipe.ensemble.detect(frame, 0.0)
if guns:
    g = guns[0]
    gx1, gy1, gx2, gy2 = g['bbox']
    crop = frame[gy1:gy2, gx1:gx2]
    
    # Check what MobileNet predicts in detail
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(rgb_crop)
    tensor = pipe.vector_verify.transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = pipe.vector_verify.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    
    top5 = torch.topk(probs, 5)
    print("Top 5 ImageNet classes for this crop:")
    for prob, idx in zip(top5.values, top5.indices):
        idx = int(idx)
        print(f"  {pipe.vector_verify.categories[idx]}: {prob:.4f}")
    
    firearm_prob = float(torch.sum(probs[pipe.vector_verify.firearm_indices]))
    print(f"Total firearm prob: {firearm_prob:.4f}")
cap.release()
