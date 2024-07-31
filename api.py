import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Model yükleme
model_path = 'my_model.keras'
model = load_model(model_path)

# Optimal threshold'u yükleme
with open('threshold.txt', 'r') as f:
    final_threshold = float(f.read())
    
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

        if result > 0.41242557764053345:
            prediction_text = "Su sayacıdır"
        else:
            prediction_text = "Su sayacı değildir"

        return jsonify({
            'prediction': prediction_text,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)