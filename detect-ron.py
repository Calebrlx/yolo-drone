import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Step 1: Load Detectron2 Configuration and Pretrained Model
cfg = get_cfg()
cfg.merge_from_file("./configs/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = (
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

# Step 2: Load the Image
image_path = "pic01.png"  # Ensure pic01.png is in the same directory
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 3: Initialize Predictor
predictor = DefaultPredictor(cfg)

# Step 4: Perform Object Detection
outputs = predictor(image)

# Debug: Print Raw Detection Results
print("Detection Outputs:", outputs["instances"].to("cpu"))

# Step 5: Visualize Results
v = Visualizer(image_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save Annotated Image
output_path = "output_detected.png"
cv2.imwrite(output_path, cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR))
print(f"Annotated image saved to {output_path}")