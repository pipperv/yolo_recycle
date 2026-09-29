import argparse

import torch
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n-seg.pt",
                        help="modelo de partida: base de Ultralytics (se descarga solo) o un .pt propio")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--name", default="pat-papercup-seg")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    model = YOLO(args.model)
    model.train(data="data.yaml",
                epochs=args.epochs,
                imgsz=640,
                device=device,
                name=args.name,
                freeze=args.freeze,
                batch=args.batch,
                workers=args.workers)
