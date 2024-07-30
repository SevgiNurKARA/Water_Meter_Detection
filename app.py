import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import logging
import sys

# Configure logging to handle non-ASCII characters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        logger.info(f"Received file: {file.filename}")

        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed types are png, jpg, jpeg"}), 400

        model_path = request.form.get('model_path', 'my_model.keras')
        logger.info(f"Original model path: {model_path}")

        safe_model_path = model_path.encode('ascii', 'ignore').decode('ascii')
        logger.info(f"Safe model path: {safe_model_path}")

        if not os.path.exists(safe_model_path):
            return jsonify({"error": "Model not found. Ensure the model file exists."}), 400

        model = load_model(safe_model_path)
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")

        # Encode the filename to avoid encoding issues
        safe_filename = file.filename.encode('ascii', 'ignore').decode('ascii')
        logger.info(f"Processing file: {safe_filename}")

        image = Image.open(file.stream)
        logger.info(f"Image opened successfully. Size: {image.size}, Mode: {image.mode}")

        # Print original image shape
        original_shape = np.array(image).shape
        logger.info(f"Original image shape: {original_shape}")

        image = image.resize((150, 150))
        logger.info("Image resized to 150x150")

        image = np.array(image)
        logger.info(f"Image converted to numpy array. Shape after resize: {image.shape}")

        if len(image.shape) == 2:  # Convert grayscale to RGB
            image = np.stack((image,) * 3, axis=-1)
            logger.info("Grayscale image converted to RGB")

        if image.shape[-1] != 3:
            logger.error(f"Unexpected number of channels: {image.shape[-1]}")
            return jsonify({"error": "Image must have 3 channels (RGB)"}), 400

        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        logger.info(f"Image preprocessed. Final shape: {image.shape}")

        if image.shape[1:] != (150, 150, 3):
            logger.error(f"Image shape {image.shape[1:]} does not match expected input shape (150, 150, 3)")
            return jsonify({"error": f"Image shape {image.shape[1:]} does not match expected input shape (150, 150, 3)"}), 400

        predictions = model.predict(image)
        logger.info(f"Predictions made: {predictions}")
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        if predicted_class_index == 0:
            result = "This image contains a counter."
        else:
            result = "This image does not contain a counter."

        logger.info(f"Prediction result: {result}")
        return jsonify({"prediction": result})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)