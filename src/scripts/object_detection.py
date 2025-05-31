from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')  # You can use yolov8s or yolov8m for better accuracy

# Path to your generated image
image_path = "data/generated_images/vide/flux_kontext_max/2025-05-31/output_16-43-01.png"

# Output folder for crops
output_dir = 'data/crops'
# ======================
exists = os.path.exists(output_dir)

if not exists:
    os.makedirs(output_dir)
    print("Daily cropped objects directory created!")


# Run inference
results = model(image_path)[0]

# Load image with OpenCV
image = cv2.imread(image_path)

# Loop through detections
for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = image[y1:y2, x1:x2]
    crop_path = os.path.join(output_dir, f"object_{i}.png")
    
    cv2.imwrite(crop_path, crop)
    print(f"Cropped object saved to: {crop_path}")

        # Crop the object
    cropped = image[y1:y2, x1:x2]
