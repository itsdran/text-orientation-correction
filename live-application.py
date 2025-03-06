import os
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
ANGLES = list(range(0, 360, 25))
MODEL_PATH = "angle_detector_model.h5"

# Load Model
model = keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess image for model inference."""
    return cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_NEAREST) / 255.0

def predict_angle(image):
    """Predict text angle in the image."""
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    return ANGLES[np.argmax(predictions)]

def rotate_image(image, angle):
    """Rotate image to make text upright."""
    if angle == 0:
        return image  # Skip rotation if already upright
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_angle = -angle

    return cv2.warpAffine(image, cv2.getRotationMatrix2D(center, rotation_angle, 1.0), (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def draw_text(image, text, position=(10, 10), font_scale=1, color=(0, 255, 0), thickness=2):
    """Overlay text on the image."""
    return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_angle_line(image, angle, color=(0, 0, 255)):
    """Draws a line indicating the detected angle on the image."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Convert angle to radians for trigonometry
    rad = np.deg2rad(angle)

    # Define line length
    line_length = min(w, h) // 3

    # Compute line endpoints
    x_end = int(center[0] + line_length * np.cos(rad))
    y_end = int(center[1] + line_length * np.sin(rad))

    # Draw line and circle at center
    image = cv2.line(image, center, (x_end, y_end), color, 2)
    image = cv2.circle(image, center, 5, color, -1)
    
    return image

# Open webcam
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Predict Angle
    predicted_angle = predict_angle(frame)

    # Rotate to Correct Orientation
    rotated_frame = rotate_image(frame, predicted_angle)

    # Draw angle line on raw image
    frame_with_angle = draw_angle_line(frame.copy(), predicted_angle)

    # Draw predicted angle text on raw image
    frame_with_text = draw_text(frame_with_angle, f"Angle: {predicted_angle} deg", (10, 50))

    # Draw angle line on rotated image (should be at 0°)
    rotated_frame_with_angle = draw_angle_line(rotated_frame.copy(), 0)

    # Draw "Rotated to 0°" text on corrected image
    rotated_frame_with_text = draw_text(rotated_frame_with_angle, "Rotated to 0 deg", (10, 50))

    # Display both original and corrected images
    combined_display = np.hstack((frame_with_text, rotated_frame_with_text))

    cv2.imshow("Live Feed - Original (with angle) vs Rotated", combined_display)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
