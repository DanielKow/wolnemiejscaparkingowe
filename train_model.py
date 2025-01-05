from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='parking_slots.yaml',
    epochs=50,
    imgsz=640,
    device="mps",
    optimizer="Adam",
    lr0=0.00025
)