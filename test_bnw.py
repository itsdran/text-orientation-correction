#Tests out the model angle_detector_model_bnw.h5

import os
import cv2
import numpy as np
import tensorflow as tf
import time
from keras.models import load_model

# Optimize GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Constants
IMG_SIZE = 128
MODEL_PATH = "angle_detector_model_bnw.h5"
TEST_IMAGE_PATH = "C:/Users/drflo/Downloads/sample/1.jpg"
OUTPUT_IMAGE_PATH = "C:/Users/drflo/Downloads/sample/output.jpg"

# Load the trained model
model = load_model(MODEL_PATH)

ANGLES = list(range(0, 360, 25))

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return bgr_img / 255.0  # Normalize to [0,1]

def predict_angle(image_path, model):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img, batch_size=1)[0]
    predicted_index = np.argmax(prediction)
    
    # Get the predicted angle
    predicted_angle = ANGLES[predicted_index] if predicted_index < len(ANGLES) else None
    confidence = prediction[predicted_index]

    return predicted_angle, confidence

def rotate_image(image, angle):
    """Rotates the image and ensures it remains in RGB format."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    angle = -(angle if angle <= 180 else 360 - angle)  # Correct rotation angle
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(center, angle, 1.0), (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

# Start timer
start_time = time.time()  

# Run the prediction
predicted_angle, confidence = predict_angle(TEST_IMAGE_PATH, model)

# Load image and apply rotation
image = cv2.imread(TEST_IMAGE_PATH)
rotated_image = rotate_image(image, predicted_angle)

# Save the rotated image
cv2.imwrite(OUTPUT_IMAGE_PATH, rotated_image)

# End timer
elapsed_time = time.time() - start_time  

print(f"Predicted angle: {predicted_angle}° with confidence: {confidence:.2f}")
print(f"Time elapsed: {elapsed_time:.4f} seconds")
print(f"Rotated image saved to: {OUTPUT_IMAGE_PATH}")