from flask import Flask, render_template, request, jsonify, send_file
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import torch
import cv2
import numpy as np
from base64 import b64encode
from io import BytesIO
import time

app = Flask(__name__)

# Load models for both captioning and question answering
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

processor_vqa = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def generate_caption(image):
    with Image.open(image) as img:
        raw_image = img.convert("RGB")

        # Use a prompt for generating longer descriptions
        prompt = "Caption:"
        inputs = caption_processor(raw_image, text=prompt, return_tensors="pt")

        start_time = time.time()
        out = caption_model.generate(**inputs, max_length=150, max_new_tokens=150)  # Fixed max_length
        generation_time = time.time() - start_time

        caption = caption_processor.decode(out[0], skip_special_tokens=True)

        # Split the caption into multiple lines for readability
        formatted_caption = caption.replace(". ", ".\n")

        return formatted_caption, generation_time

def convert_image_to_base64(image):
    """Converts an image to Base64 format for display."""
    pil_image = Image.open(image).convert('RGB')
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_data = b64encode(buffered.getvalue()).decode('utf-8')

    return img_data

def answer_question(image, question):
    with Image.open(image) as img:
        raw_image = img.convert("RGB")

        inputs = processor_vqa(raw_image, question, return_tensors="pt")
        out = model_vqa.generate(**inputs, max_length=50)  # Add max_length to avoid long responses

        # Use caption_processor to decode, not processor_vqa
        answer = caption_processor.decode(out[0], skip_special_tokens=True)
        return answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']

        try:
            image_base64 = convert_image_to_base64(image)
        except Exception:
            return render_template('index.html', generation_message="Error processing image")
        
        try:
            caption, generation_time = generate_caption(image)
        except Exception:
            return render_template('index.html', generation_message="Error generating caption")

        generation_message = f"Generated in {generation_time:.2f} seconds" if generation_time is not None else "Generated in -.-- seconds"

        return render_template('index.html', image=image_base64, caption=caption, generation_message=generation_message)
    
    return render_template('index.html')

@app.route('/api/generate_caption', methods=['POST'])
def generate_caption_api():
    if 'image' in request.files:
        image = request.files['image']
        caption, _ = generate_caption(image)
        image_name = image.filename

        response = {
            'image_name': image_name,
            'description': caption
        }

        return jsonify(response)
    else:
        return jsonify({'error': 'No image uploaded'})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    if 'image' in request.files and 'question' in request.form:
        image = request.files['image']
        question = request.form['question']
        
        try:
            answer = answer_question(image, question)
            return jsonify({'question': question, 'answer': answer})
        except Exception as e:
            return jsonify({'error': 'Error processing question', 'details': str(e)})
    
    return jsonify({'error': 'Image and question required'})

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageEnhance
import os

UPLOAD_FOLDER = 'uploads'
ENHANCED_FOLDER = 'enhanced'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

@app.route('/enhance')
def enhance_image():
    return render_template('enhance.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded!"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file!"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Enhance Image
    image = Image.open(file_path)
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(2)  # Enhance contrast

    enhanced_path = os.path.join(ENHANCED_FOLDER, file.filename)
    enhanced_image.save(enhanced_path)

    return jsonify({
        "success": True,
        "enhanced_url": f"/enhanced/{file.filename}"
    })

@app.route('/enhanced/<filename>')
def get_enhanced_image(filename):
    return send_from_directory(ENHANCED_FOLDER, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
