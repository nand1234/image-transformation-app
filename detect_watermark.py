import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



def detect_watermark(image_array):
    # Create a directory for weights if it doesn't exist
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    # Load model and processor with specified cache directory
    processor = AutoImageProcessor.from_pretrained("amrul-hzz/watermark_detector", cache_dir=weights_dir)
    model = AutoModelForImageClassification.from_pretrained("amrul-hzz/watermark_detector", cache_dir=weights_dir)

    # Load and preprocess the image

    # Use the processor to prepare the image
    inputs = processor(images=image_array, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class and logits
    logits = outputs.logits
    predicted_class = logits.argmax(dim=1).item()
    
    return predicted_class, image_array

if __name__ == '__main__':
    # Optionally display the image
    file = 'input_images/watermark.jpeg'
    img_array = np.array(Image.open(file))
    predicted_class, image = detect_watermark(img_array)
    print(f"function output: {predicted_class}")
    plt.imshow(image)
    plt.axis('off')
    plt.title("Detected Watermark" if predicted_class == 1 else "No Watermark Detected")
    plt.show()
