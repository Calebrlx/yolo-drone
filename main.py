import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Step 1: Load the YOLOv8 Model
# Ensure the "ultralytics" library is installed (pip install ultralytics)
model = YOLO("yolov8n.pt")  # Lightweight YOLOv8 model. Replace with "yolov8x.pt" for higher accuracy.

# Step 2: Load the Image
image_path = "./pic01.png"  # Replace with the path to your PNG image
output_path = "output_pic01.png"  # Path to save the image with detections

# Step 3: Run Object Detection
results = model(image_path)

# Step 4: Save the Detected Image
# YOLO's `results.save()` saves the output with bounding boxes
results.save(save_dir="./")  # Saves results in the current directory with bounding boxes

# Optional: Display the Detected Image
# Load the saved image and display it
detected_image = cv2.imread(output_path)
detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
plt.imshow(detected_image)
plt.axis("off")  # Hide axes
plt.title("Object Detection Results")
plt.show()