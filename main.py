import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Lightweight YOLOv8 model

# Load the image
image_path = "pic01.png"  # Path to your PNG image
output_path = "output_detected.png"  # Path to save the result

# Run inference
results = model(image_path)

# Process results
for result in results:  # Iterate through predictions
    # Draw bounding boxes on the original image
    image = cv2.imread(image_path)
    for box in result.boxes.data:  # Boxes data includes coordinates and confidence
        x1, y1, x2, y2, conf, cls = box.tolist()  # Extract box coordinates and info
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
        label = f"{model.names[int(cls)]} {conf:.2f}"  # Class name and confidence

        # Draw the box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the annotated image
    cv2.imwrite(output_path, image)

print(f"Detection results saved to {output_path}")


