from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
# data: path to dataset.yaml
# epochs: number of training epochs
# imgsz: image size
if __name__ == '__main__':
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        device=0, # Use 0 for GPU if available, or 'cpu'
        batch=16,
        name='fish_detector'
    )
    
    print("Training completed. Best model saved in runs/detect/fish_detector/weights/best.pt")
