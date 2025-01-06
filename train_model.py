from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(
    data='parking_slots.yaml',
    epochs=50,
    imgsz=640,
    device="mps"
)