import cv2
import numpy as np

# Function to create a test image
def create_test_image():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)  # White square
    cv2.line(img, (150, 100), (150, 50), (0, 0, 255), 3)  # Red line pointing up
    return img

# Function to rotate an image by a given angle
def rotate_frame(frame: np.ndarray, angle: float) -> np.ndarray:
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return rotated

# Define an input angle (this is the "raw angle" of the object)
angle = 360  # Try different angles (e.g., 150, 200, etc.)

rotation_angle = -angle  # Rotate counterclockwise (left to right)

# Create the raw tilted image
raw_img = create_test_image()
tilted_img = rotate_frame(raw_img, angle)  # Simulating initial angle

# Rotate the tilted image back to upright using your logic
upright_img = rotate_frame(tilted_img, rotation_angle)

# Concatenate the two views side by side
combined = np.hstack((tilted_img, upright_img))

# Display the result
cv2.imshow(f"Left: Raw Angle ({angle}°) | Right: Rotated ({rotation_angle}°)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
