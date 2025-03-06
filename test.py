#Tests out the model angle_detector_model.h

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Optimize GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Constants
IMG_SIZE = (128, 128)
ANGLES = list(range(0, 360, 25))
MODEL_PATH = "angle_detector_model.h5"
IMAGE_PATH = "C:/Users/drflo/Downloads/sample/7.jpg"
OUTPUT_FOLDER = "C:/Users/drflo/Downloads/sample"

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

def predict_text_angle(image):
    """Predict the text angle using the trained model."""
    img = cv2.resize(image, IMG_SIZE) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    angle_index = np.argmax(predictions)
    return ANGLES[angle_index] if angle_index < len(ANGLES) else None

def rotate_image(image, angle):
    """Rotate the image to correct its orientation."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    angle = -(angle if angle <= 180 else 360 - angle)
    return cv2.warpAffine(image, cv2.getRotationMatrix2D(center, angle, 1.0), (w, h), borderMode=cv2.BORDER_REPLICATE)

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Start timer
start_time = time.time()

# Load and process image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Error: Cannot read image {IMAGE_PATH}")

predicted_angle = predict_text_angle(image)
rotated_image = rotate_image(image, predicted_angle)
output_path = os.path.join(OUTPUT_FOLDER, f"output.jpg")
cv2.imwrite(output_path, rotated_image)
print(f"\t\t Predicted Angle: {predicted_angle}° \t\t Elapsed Time: {time.time() - start_time:.4f} sec")


# Notes:
# Normal - 2.2s
# DFI and Grayscaling dataset w/o preprocessing - 2.3s
# DFI and Grayscaling dataset w/ preprocessing - 2.4s
# No DFI and Grayscaling dataset w/ preprocessing - 2.7s