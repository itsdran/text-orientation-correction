import os
import time
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Disable unnecessary logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Optimize TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Constants
IMG_SIZE = (128, 128)
DISPLAY_SIZE = (300, 300)  # Adjusted display size for visualization
ANGLES = list(range(0, 360, 25))
MODEL_PATH = "angle_detector_model.h5"
IMAGE_FOLDER = "C:/Users/drflo/Downloads/cropped_images"

# Load Model
model = keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess image for model inference."""
    return cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_NEAREST) / 255.0

def predict_angle(image):
    """Predict angle of a single image."""
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    return ANGLES[np.argmax(predictions)]

def rotate_image(image, angle):
    """Rotate image to correct text orientation."""
    if angle > 350 or angle < 25 or angle == 175:
        return image  # Skip rotation if angle is within these thresholds
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    angle = -(angle if angle <= 180 else 360 - angle)  # Adjust rotation
    return cv2.warpAffine(image, cv2.getRotationMatrix2D(center, angle, 1.0), (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def draw_text(image, text, position=(10, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    """Overlay text on the image."""
    return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# Select four random images
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if len(image_files) < 4:
    raise FileNotFoundError("Not enough images in the directory.")

selected_images = random.sample(image_files, 4)

# Process each image
elapsed_times = []
comparisons = []

for img_name in selected_images:
    image_path = os.path.join(IMAGE_FOLDER, img_name)

    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {img_name}")
        continue

    # Start Timer
    start_time = time.time()

    # Predict Angle & Rotate Image
    predicted_angle = predict_angle(image)
    rotated_image = rotate_image(image, predicted_angle)

    # Resize both images for display
    image_resized = cv2.resize(image, DISPLAY_SIZE)
    rotated_resized = cv2.resize(rotated_image, DISPLAY_SIZE)

    # Draw angle on original image
    image_with_text = draw_text(image_resized, f"Angle: {predicted_angle}")

    # Concatenate images for side-by-side comparison
    comparison = np.hstack((image_with_text, rotated_resized))
    comparisons.append(comparison)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_times.append(elapsed_time)

    print(f"Image: {img_name} | Predicted Angle: {predicted_angle}")

# Stack all four image comparisons vertically
final_display = np.vstack(comparisons)

# Show Images
cv2.imshow("Original vs Rotated (4 Images)", final_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print total elapsed time
print(f"Total Processing Time for 4 Images: {sum(elapsed_times):.4f} sec")
