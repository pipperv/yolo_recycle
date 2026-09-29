import torch
from ultralytics import YOLO

if __name__ == '__main__':
    torch.cuda.empty_cache()
    model = YOLO("models/pat-papercup-seg.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="data.yaml",
                          epochs=200,
                          imgsz=640, device='cuda',
                          name="pat-papercup-seg_v2",
                          freeze=10,
                          batch=16,
                          workers=2)