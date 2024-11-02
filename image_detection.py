
import os
import urllib.request
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

YOLO_Model_name = 'yolo11x.pt'

def download_weights(url, filename):
    weights_dir = 'weights'
    weights_path = os.path.join(weights_dir, filename)
    
    if not os.path.exists(weights_path):
        print(f"Downloading {filename}...")
        os.makedirs(weights_dir, exist_ok=True)
        
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, weights_path, reporthook=lambda b, bsize, tsize: t.update(bsize))
        
        print(f"{filename} downloaded successfully.")
    
    return weights_path

def download_yolo_weights():
    weights_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{YOLO_Model_name}"
    return download_weights(weights_url, YOLO_Model_name)

def detect_objects(img_array):
    download_yolo_weights()
    model = YOLO(f"weights/{YOLO_Model_name}")  # Ensure this path is correct

    # Perform inference
    results = model(img_array)

    # Process results
    detections = results[0].boxes
    print(f"Total detections: {len(detections)}")

    if len(detections) == 0:
        print("No objects detected.")
        return img_array, []

    # Convert numpy array to PIL Image for drawing
    image = Image.fromarray(img_array)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Sort detections by confidence
    sorted_detections = sorted(detections, key=lambda x: x.conf, reverse=False)

    detection_results = []
    print("\nall detections:")
    for i, detection in enumerate(sorted_detections, 1):
        box = detection.xyxy[0].tolist()
        class_id = int(detection.cls)
        confidence = detection.conf.item()
        class_name = model.names[class_id]

        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{class_name}: {confidence:.2f}", font=font, fill="red")
        print(f"{i}. {class_name}: {confidence:.2f}")
        
        detection_results.append({
            'class': class_name,
            'confidence': confidence,
            #'box': box
        })

    return np.array(image), detection_results

if __name__ == '__main__':
    file = 'input_images/watermark.jpeg'
    img_array = np.array(Image.open(file))
    detect_objects(img_array)
