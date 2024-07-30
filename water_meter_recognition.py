import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_safely(model_path):
    if os.path.exists(model_path):
        logger.info(f"Model file found: {os.path.abspath(model_path)}")
        try:
            model = load_model(model_path)
            logger.info("Model loaded successfully.")
            model.summary()  # Print model summary to verify input shape
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    else:
        logger.error(f"Model file not found: {os.path.abspath(model_path)}")
        return None

def evaluate_model(model, test_images, test_labels):
    datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen.flow(test_images, test_labels, batch_size=32, shuffle=False)

    logger.info(f"Test images shape: {test_images.shape}")
    logger.info(f"Test labels shape: {test_labels.shape}")

    try:
        test_loss, test_accuracy = model.evaluate(test_generator)
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")

        predictions = model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)

        print("\nClassification Report:")
        print(classification_report(test_labels, predicted_classes))

        cm = confusion_matrix(test_labels, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Image could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))  # Adjusted to match training size
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def predict_image(model, image):
    try:
        prediction = model.predict(image)
        return np.argmax(prediction)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def process_images(model, test_images_folder):
    results = []
    for category in os.listdir(test_images_folder):
        category_path = os.path.join(test_images_folder, category)
        if os.path.isdir(category_path):
            print(f"Processing: {category_path}")
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                if os.path.isfile(image_path):
                    print(f"  Processing image: {image_path}")
                    image = preprocess_image(image_path)
                    if image is not None:
                        result = predict_image(model, image)
                        if result is not None:
                            result_label = "Image is a water meter." if result == 1 else "Image is not a water meter."
                            results.append((image_name, result_label))
                            print(f"  {image_name}: {result_label}")
                else:
                    print(f"  Error: {image_path} is not a file.")
        else:
            print(f"Error: {category_path} is not a directory.")
    return results

def main():
    model_path = 'my_model.keras'
    model = load_model_safely(model_path)
    if model is None:
        return

    test_images_folder = 'test'
    if not os.path.exists(test_images_folder):
        print(f"Error: {test_images_folder} directory not found.")
        return

    print("Files in the test folder:")
    for file in os.listdir(test_images_folder):
        print(file)

    results = process_images(model, test_images_folder)

    with open('results.txt', 'w') as f:
        for image_name, label in results:
            f.write(f"{image_name}: {label}\n")

    print("Prediction results have been saved to 'results.txt'.")

if __name__ == "__main__":
    main()
