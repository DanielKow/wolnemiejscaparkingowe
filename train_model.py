from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='parking_slots.yaml', epochs=50, imgsz=640, device="mps")
# with increased confidence threshold and data augmentation
# model.train(
#     data='parking_slots.yaml',
#     epochs=50,
#     imgsz=640,
#     device="mps",
#     conf=0.4,  # Confidence threshold
#     augment=True  # Enable data augmentation
# )