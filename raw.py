from ultralytics import YOLO

# Step 1: Load the YOLOv8 Model
model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 model

# Step 2: Load the Image
image_path = "pic01.png"

# Step 3: Run Object Detection
results = model(image_path, conf=0.25)

# Step 4: Output Raw Detection Data
for result in results:  # Iterate through results (one per image)
    for box in result.boxes.data:  # Iterate through detections
        x1, y1, x2, y2, conf, cls = box.tolist()  # Extract detection details
        class_name = model.names[int(cls)]  # Map class index to name
        print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")