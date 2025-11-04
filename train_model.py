from ultralytics import YOLO

# Load YOLOv8 small model (lightweight and fast)
model = YOLO('yolov8s.pt')

# Train the model
model.train(
    data='data.yaml',  # your dataset configuration
    epochs=50,         # number of times the model sees your dataset
    imgsz=640,         # image size
    batch=16,          # how many images per training batch
)
