import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "models" / "pat-papercup-seg.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=MODEL_PATH)
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--conf", type=float, default=0.75)
args = parser.parse_args()

if not args.model.is_file():
    raise SystemExit(f"No se encontró el modelo: {args.model}. Descárgalo como indica el README.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading YOLO model on {device}...")
model = YOLO(args.model)

for result in model(source=args.camera, stream=True, conf=args.conf, device=device, verbose=False):
    cv2.imshow("YOLO Webcam", result.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
