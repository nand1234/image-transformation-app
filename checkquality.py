import cv2
from PIL import Image
import numpy as np


# Function to calculate image sharpness
def calculate_sharpness(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use the Laplacian method to calculate the variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# Function to analyze image composition
def analyze_composition(image):
    # Calculate the number of channels
    channels = image.shape[2] if len(image.shape) == 3 else 1
    # Calculate the dominant color in BGR format
    mean_color = cv2.mean(image)[:3]  # Get BGR mean color
    return channels, mean_color

# Function to check for visible watermarks
def detect_image_quality(img_array):
    # Get image resolution
    height, width = img_array.shape[:2]
    resolution = f"{width} x {height}"

    # Analyze composition
    channels, mean_color = analyze_composition(img_array)

    # Calculate sharpness
    sharpness = calculate_sharpness(img_array)
    
    # return image analysis results
    return resolution, channels, mean_color, sharpness

if __name__ == '__main__':
    # Replace 'image_path.jpg' with the path to your image
    file = 'input_images/watermark.jpeg'
    img_array = np.array(Image.open(file))
    resolution, channels, mean_color, sharpness = detect_image_quality(img_array)
    print(f"Resolution: {resolution}")
    print(f"Channels: {channels}")
    print(f"Mean Color (BGR): {mean_color}")
    print(f"Sharpness (Laplacian Variance): {sharpness}")
