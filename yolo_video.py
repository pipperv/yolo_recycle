import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "models" / "pat-papercup-seg.pt"
VIDEOS_DIR = Path(__file__).parent / "videos"
MIN_AREA = 20000

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", default="papercup_light.mp4", help="ruta o nombre de un archivo en videos/")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--step", action="store_true", help="avanzar frame a frame con cualquier tecla")
args = parser.parse_args()

video_path = Path(args.video)
if not video_path.is_file():
    video_path = VIDEOS_DIR / args.video
if not video_path.is_file():
    raise SystemExit(f"No se encontró el video: {args.video}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading YOLO model on {device}...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(str(video_path))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    for r in model(frame, conf=args.conf, device=device, verbose=False):
        if r.masks is None:
            continue
        for m in r.masks:
            if m.data.sum().item() < MIN_AREA:
                continue
            mask = cv2.resize(m.data.cpu().numpy().squeeze(), (w, h))
            ys, xs = np.nonzero(mask > 0.5)
            cv2.circle(frame, (int(xs.mean()), int(ys.mean())), radius=10, color=(255, 0, 255), thickness=-1)
            for a, b in m.xy[0].tolist():
                cv2.circle(frame, (int(a), int(b)), radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow("YOLO Inference", frame)
    if cv2.waitKey(0 if args.step else 1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
