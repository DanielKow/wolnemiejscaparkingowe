from click.core import batch
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='parking_slots.yaml',
    epochs=50,
    imgsz=640,
    device="mps",
    batch=8,
    conf=0.3,         # Increase confidence threshold
    max_det=200       # Decrease from the default (300)
)