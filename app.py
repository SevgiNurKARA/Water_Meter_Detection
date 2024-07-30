from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from flask_cors import CORS
from water_meter_recognition import predict_image
import tempfile
import imghdr
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from logging.handlers import RotatingFileHandler

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def load_model_safely(model_path):
    if os.path.exists(model_path):
        logger.info(f"Model file found: {os.path.abspath(model_path)}")
        try:
            model = load_model(model_path)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    else:
        logger.error(f"Model file not found: {os.path.abspath(model_path)}")
        return None

def preprocess_image(image_path):
    logger.debug(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        logger.error(f"File not found: {image_path}")
        return None

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to read image.")
        logger.debug(f"Image read, shape: {image.shape}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.debug(f"Image color converted to RGB, shape: {image.shape}")

        image = cv2.resize(image, (224, 224))
        logger.debug(f"Image resized, new shape: {image.shape}")

        image = np.expand_dims(image, axis=0)
        logger.debug(f"Image dimensions expanded, new shape: {image.shape}")

        image = image / 255.0
        logger.debug(f"Image normalized, min: {image.min()}, max: {image.max()}")

        return image
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        return None

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
    app.config['MODEL'] = load_model_safely('my_model.keras')

    Talisman(app, content_security_policy=None)

    limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
    limiter.init_app(app)

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File is too large. Maximum size is 16 MB."}), 413

    @app.route('/predict', methods=['POST'])
    @limiter.limit("10 per minute")
    def predict_image_api():
        logger.info("Received a request")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request files: {request.files}")

        if request.content_type != 'multipart/form-data':
            logger.error("Incorrect content type: %s", request.content_type)
            return jsonify({'error': 'Request must be multipart/form-data'}), 400

        if 'image' not in request.files:
            logger.error("No 'image' key in request.files")
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            logger.error("File name is empty")
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            ext = validate_image(file.stream)
            if not ext:
                logger.error("Invalid image file")
                return jsonify({'error': 'Invalid image file'}), 400
        else:
            logger.error("File type not allowed")
            return jsonify({'error': 'File type not allowed'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        logger.debug(f"Temporary file path: {temp_path}")

        # Process the image and make predictions
        image = preprocess_image(temp_path)
        if image is None:
            logger.error("Failed to process image")
            return jsonify({'error': 'Failed to process image'}), 400

        logger.debug(f"Processed image shape: {image.shape}")

        try:
            model = app.config['MODEL']
            result = predict_image(model, image)
            if result is None:
                logger.error("Failed to make prediction")
                return jsonify({'error': 'Failed to make prediction'}), 500
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return jsonify({'error': 'Error during prediction'}), 500

        result_label = "Görsel bir su sayacıdır." if result == 1 else "Görsel bir su sayacı değildir."
        logger.info(f"Prediction result: {result_label}")

        # Remove the temporary file
        try:
            os.unlink(temp_path)
            logger.debug("Temporary file removed successfully")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {str(e)}")

        return jsonify({'prediction': result_label})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
