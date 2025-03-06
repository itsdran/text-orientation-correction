import os
import cv2
import numpy as np

# Paths
input_folder = "dataset/000"
output_base = "dataset"

# Ensure output folders exist
rotation_angles = list(range(0, 360, 25))
for angle in rotation_angles:
    os.makedirs(os.path.join(output_base, f"{angle:03}"), exist_ok=True)

def detect_and_crop_text(image_path):
    """ Detects text and crops it. Returns None if no text is found. """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection to detect potential text regions
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours based on edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, delete the image
    if not contours:
        os.remove(image_path)
        print(f"Deleted {image_path} (no text detected)")
        return None

    # Find bounding box for the text area
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    
    # Crop text region
    cropped = img[y:y+h, x:x+w]
    return cropped

def rotate_and_save(image, image_name, angle):
    """ Rotates the cropped text image and saves it in the correct folder. """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix and rotate image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Save rotated image in the respective folder
    save_path = os.path.join(output_base, f"{angle:03}", image_name)
    cv2.imwrite(save_path, rotated)
    print(f"Saved rotated image: {save_path}")

def process_images():
    """ Processes images: detects text, crops, and rotates. """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_folder, filename)

            # Detect and crop text
            cropped = detect_and_crop_text(image_path)
            if cropped is None:
                continue  # Skip if no text was detected

            # Save cropped image in input folder (overwrite original)
            cropped_path = os.path.join(input_folder, filename)
            cv2.imwrite(cropped_path, cropped)

            # Rotate and save the cropped image at different angles
            for angle in rotation_angles:
                rotate_and_save(cropped, filename, angle)

if __name__ == "__main__":
    process_images()
    print("Processing complete!")
