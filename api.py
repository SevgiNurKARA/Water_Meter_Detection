import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Model yükleme
model_path = 'my_model.keras'
model = load_model(model_path)

def preprocess_image(image):
    image = Image.open(io.BytesIO(image))
    image = image.convert('RGB')
    image = image.resize((150, 150))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = image_file.read()

    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        result = prediction[0][0]

        is_water_meter = result > 0.5
        confidence = float(result) if is_water_meter else float(1 - result)

        return jsonify({
            'is_water_meter': bool(is_water_meter),
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)