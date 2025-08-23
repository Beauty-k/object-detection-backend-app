from ultralytics import YOLO

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    model.train(
        data="custom_dataset/scale/data.yaml",
        epochs=50,
        imgsz=640
    )