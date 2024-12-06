from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Use yolov8x.pt for better results if needed

# Load the image
image_path = "pic01.png"
results = model(image_path, conf=0.1)  # Lower confidence threshold for more detections

# Debugging: Check model classes
print(f"Model Classes: {model.names}")

# Output raw detection data
if results[0].boxes.data.shape[0] == 0:  # No detections
    print("No detections found.")
else:
    for box in results[0].boxes.data:  # Iterate through detections
        x1, y1, x2, y2, conf, cls = box.tolist()  # Extract detection details
        class_name = model.names[int(cls)]  # Map class index to name
        print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

# Optional: Save a debug image to visually confirm input
image = cv2.imread(image_path)
cv2.imwrite("debug_input.png", image)