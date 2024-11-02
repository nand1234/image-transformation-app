import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import logging
import os
import numpy as np

# Suppress unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Directory for model weights
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)

# Initialize model and processor
def load_model_and_processor():
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", cache_dir=weights_dir)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", cache_dir=weights_dir)
    model.eval().to("cpu")
    return model, processor

# Preprocess the image array for model input
def preprocess_image(image_array, processor):
    image = Image.fromarray(image_array)  # Convert the NumPy array to a PIL Image
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    return image, inputs

# Run inference and post-process results
def run_inference(model, processor, inputs, image):
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    return results

# Draw bounding boxes on the image and return predictions
def draw_boxes_on_image(image, results, model):
    draw = ImageDraw.Draw(image)
    predictions = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        score = round(score.item(), 3)
        
        # Draw box on image
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{label_name} {score}", fill="red")
        
        # Append prediction to list
        predictions.append({"label": label_name, "score": score, "box": box})
    
    return predictions, image

# Extract labels and scores from predictions
def extract_labels_and_scores(predictions):
    return [{"label": pred["label"], "score": pred["score"]} for pred in predictions]

# Main function to run all steps using an image array as input
def detect(image_array):
    model, processor = load_model_and_processor()
    image, inputs = preprocess_image(image_array, processor)
    results = run_inference(model, processor, inputs, image)
    predictions, output_image = draw_boxes_on_image(image, results, model)
    
    # Extract labels and scores only
    labels_and_scores = extract_labels_and_scores(predictions)
    
    # Save or display the output image with boxes
    #output_image.save("output_with_boxes.jpg")
    
    return np.array(image), labels_and_scores

# Example of using the main function with an image array
# Assuming 'image_array' is a NumPy array representation of an image
if __name__=='__main__':
    image_path = "input_images/test.jpeg"
    image_array = np.array(Image.open(image_path))  # Load and convert to array
    labels_and_scores = detect(image_array)
    print(labels_and_scores)
