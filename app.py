import os
import base64
import io
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from run_real_esrgan import run_real_esrgan
from detect_watermark import detect_watermark
from checkquality import detect_image_quality
from image_detection import detect_objects
from run_real_esrgan import run_real_esrgan
from image_detection_resnet import detect


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    scale = int(request.form.get('scale'))  # Default to 1.0 if not provided
    print(scale)

    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Read image file into a numpy array
        img_array = np.array(Image.open(file.stream))
        
        # Process the image
        enhanced_array = run_real_esrgan(img_array, scale)
        
        
        # Convert numpy arrays to base64 encoded strings
        enhanced_base64 = array_to_base64(enhanced_array)
        
        return jsonify({
            'enhanced_image': enhanced_base64,
        })
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/check_image', methods=['POST'])
def check_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Read image file into a numpy array
        #img_array = np.array(Image.open(file.stream))
        img_array = np.array(Image.open(file.stream))

        # Detect objects
        detected_array, detections = detect(img_array)
        detected_base64 = array_to_base64(detected_array)

        # Process the image``
        predicted_class, image = detect_watermark(img_array)
        print(f"image array for detect_watermark {predicted_class}")
        
        # Detect objects
        resolution, channels, mean_color, sharpness = detect_image_quality(img_array)
                
        return jsonify({
            'detected_image': detected_base64,
            'detections': detections,
            'watermark_result': bool(predicted_class),
            'quality_detections': f"resolution:{resolution}, channels:{channels}, mean_color:{mean_color}, sharpness={sharpness}"

        })
    
    return jsonify({'error': 'Invalid file type'})


def array_to_base64(array):
    img = Image.fromarray(array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
