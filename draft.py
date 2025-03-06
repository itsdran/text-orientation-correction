import cv2
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from rembg import remove
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True)

# Constants
IMG_SIZE = (128, 128)
ANGLES = list(range(0, 360, 25)) 
DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)

def capture_image(isNewDataset):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        results = ocr.ocr(frame, cls=True)
        detected_text = None
        dup = frame.copy()

        if results and results[0]:
            for res in results[0]:
                points = np.array(res[0], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)
                detected_text = res[1][0]
                                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, detected_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1)

        if key == 32 and detected_text:
            if isNewDataset:
                folder = f"{DATASET_DIR}/{detected_text}"
            else:
                folder = f"{DATASET_DIR}"
            os.makedirs(folder, exist_ok=True)
            image_path = f"{folder}/{detected_text}.jpg"
            cv2.imwrite(image_path, dup)
            
            # Remove background and update image_path
            image_path = remove_background(image_path)
            
            cap.release()
            cv2.destroyAllWindows()
            
            return image_path, detected_text
        
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return None, None
        
def remove_background(image_path):
    img = Image.open(image_path)
    img_no_bg = remove(img)

    # Convert RGBA to RGB since JPEG doesn't support transparency
    if img_no_bg.mode == 'RGBA':
        img_no_bg = img_no_bg.convert('RGB')

    new_path = image_path.replace(".jpg", ".jpg")
    img_no_bg.save(new_path, format="JPEG")
    
    return new_path 

def generate_rotated_images(image_path, text):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    for angle in ANGLES[1:]:  # Skip 0° since it already exists
        folder = f"{DATASET_DIR}/{angle:03d}"
        os.makedirs(folder, exist_ok=True)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        save_path = f"{folder}/{text}_{angle}_deg.jpg"
        cv2.imwrite(save_path, rotated)

def create_model(num_classes):    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Preprocess dataset 
def preprocess_dataset():
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=16, subset='training', class_mode='categorical')
    val_data = datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=16, subset='validation', class_mode='categorical')
    print(f"Classes: {train_data.class_indices}")
    return train_data, val_data

def train_model():
    train_data, val_data = preprocess_dataset()
    num_angles = len(train_data.class_indices) 
    
    model = create_model(num_angles)
    model.fit(train_data, validation_data=val_data, epochs=10)
    
    model.save("angle_detector_model.h5")
    _, accuracy = model.evaluate(val_data)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    history = model.fit(train_data, validation_data=val_data, epochs=10)
    plot_precision_loss_curve(history)
    _, accuracy = model.evaluate(val_data)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
def plot_precision_loss_curve(history):
    plt.figure(figsize=(8, 6))

    # Plot accuracy (Precision) vs. loss
    plt.plot(history.history["accuracy"], label="Train Accuracy", marker='o')
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker='o')
    plt.plot(history.history["loss"], label="Train Loss", linestyle="dashed", marker='x')
    plt.plot(history.history["val_loss"], label="Validation Loss", linestyle="dashed", marker='x')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Precision-Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_text_angle(image_path):
    model = keras.models.load_model("angle_detector_model.h5")

    img = cv2.imread(image_path)
    
    # Ensure image is RGB before processing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, IMG_SIZE) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    angle_index = np.argmax(predictions)
    
    if angle_index >= len(ANGLES):
        print(f"Error: angle_index {angle_index} is out of range")
        return None, None

    predicted_angle = ANGLES[angle_index]
    confidence = np.max(predictions)

    print(f"Predicted Angle: {predicted_angle}° with Confidence: {confidence:.2f}")
    
    return predicted_angle, confidence

def upright_image(image_path):
    predicted_angle, confidence = predict_text_angle(image_path)

    if predicted_angle is None:
        print("Failed to predict angle. Returning original image.")
        return image_path

    if predicted_angle == 0:
        print("Image is already upright.")
        return image_path

    # Negate the predicted angle
    
    if predicted_angle < 180:
        correction_angle = predicted_angle - 180
    else:
        correction_angle = predicted_angle

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

    # Adjust bounding box size to prevent cropping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust transformation matrix to center the rotated image
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Apply rotation while keeping the full image
    upright_img = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

    # Save the corrected image
    corrected_path = image_path.replace(".jpg", "_upright.jpg")
    cv2.imwrite(corrected_path, upright_img)

    print(f"Image rotated by {correction_angle}° to make it upright. Saved as: {corrected_path}")
    return corrected_path

while True:
    print("\nChoose an option:\n1. Capture a new dataset\n2. Train model\n3. Predict text angle\n4. Plot Precision Loss Curve\n5. ")
    choice = input("Enter choice: ")
    
    if choice == "1":
        isNewDataset = True        
        image_path, detected_text = capture_image(isNewDataset)
        if image_path and detected_text:
            generate_rotated_images(image_path, detected_text)
    elif choice == "2":
        train_model()
    elif choice == "3":
        isNewDataset = False        
        image_path, _ = capture_image(isNewDataset)
        if image_path:
            predicted_angle, _ = predict_text_angle(image_path)
            upright_image (image_path)
    elif choice =="4":
        plot_precision_loss_curve()
    elif choice == "5":
        break
