# Streamlit app to generate an image, detect & crop objects, and show product matches

import streamlit as st
import os
import time
import json
from PIL import Image
import cv2
from ultralytics import YOLO

# Paths
GENERATED_IMAGE_PATH = "data/generated_images/vide/flux_kontext_max"
CROPPED_FOLDER = "data/generated_images/vide/flux_kontext_max/2025-05-31/cropped_objects"
MATCHES_JSON = "matched_products.json"

# Load detection model
@st.cache_resource

def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Load latest generated image
def get_latest_image():
    all_dirs = sorted(os.listdir(GENERATED_IMAGE_PATH), reverse=True)
    for d in all_dirs:
        dir_path = os.path.join(GENERATED_IMAGE_PATH, d)
        images = sorted(os.listdir(dir_path), reverse=True)
        if images:
            return os.path.join(dir_path, images[0])
    return None

# Crop detected objects
def detect_and_crop_objects(image_path):
    os.makedirs(CROPPED_FOLDER, exist_ok=True)
    image = cv2.imread(image_path)
    results = model(image_path)
    boxes = results[0].boxes
    names = model.names
    cropped_info = []

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        filename = f"{i}_{class_name}_{int(conf*100)}.jpg"
        path = os.path.join(CROPPED_FOLDER, filename)
        cv2.imwrite(path, cropped)
        cropped_info.append({"file": filename, "label": class_name})

    return cropped_info

# Load matches
@st.cache_data
def load_matches():
    if not os.path.exists(MATCHES_JSON):
        return []
    with open(MATCHES_JSON) as f:
        return json.load(f)

# UI
st.title("🛋️ AI Product Finder from Interior Design")

latest_image_path = get_latest_image()

if latest_image_path:
    st.image(latest_image_path, caption="Generated Image", use_column_width=True)

    if st.button("🔍 Detect & Find Similar Products"):
        with st.spinner("Detecting and finding matches..."):
            cropped_info = detect_and_crop_objects(latest_image_path)
            matches_data = load_matches()

        for item in cropped_info:
            filename = item["file"]
            label = item["label"]

            st.markdown(f"### 🔹 {label.title()} Detected")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(os.path.join(CROPPED_FOLDER, filename), width=150)
            with col2:
                for match_set in matches_data:
                    if match_set["source_image"] == filename:
                        for match in match_set["matches"]:
                            st.markdown(
                                f"<a href='{match['link']}' target='_blank'>"
                                f"<img src='{match['image']}' width='100'><br>"
                                f"{match['title']}</a><br><br>", unsafe_allow_html=True
                            )
else:
    st.warning("No generated image found. Please run the generation script first.")