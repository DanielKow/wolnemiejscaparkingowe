import cv2
import numpy as np
from saving_results import ResultsSaver

# Initialize ResultsSaver
saver = ResultsSaver()

# Fixed image path
image_path = 'test_images/2012-09-11_16_48_36_jpg.rf.4ecc8c87c61680ccc73edc218a2c8d7d.jpg'

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Nie można załadować obrazu. Sprawdź poprawność ścieżki.")

saver.save(image, "original_image")

# Convert to HSV for color thresholding
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
saver.save(hsv, "hsv_image")

# Threshold for bright colors (white parking lines)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])
mask_white = cv2.inRange(hsv, lower_white, upper_white)

saver.save(mask_white, "white_color_mask")

# Smooth the mask to reduce noise
smoothed_mask = cv2.medianBlur(mask_white, 5)
saver.save(smoothed_mask, "smoothed_white_mask")

# Combine with grayscale edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
saver.save(gray, "grayscale_image")

# Adaptive histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
saver.save(enhanced_gray, "enhanced_grayscale")

# Adaptive edge detection (Sobel + Canny)
edges_sobel = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X direction
edges_sobel = np.uint8(np.absolute(edges_sobel))
saver.save(edges_sobel, "sobel_edges")

# Canny edge detection (less strict thresholds)
edges_canny = cv2.Canny(enhanced_gray, 50, 120)
saver.save(edges_canny, "canny_edges")

# Combine edges and color mask
combined = cv2.bitwise_or(smoothed_mask, edges_canny)
saver.save(combined, "combined_mask")

# Use morphological operations to enhance combined result
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(combined, kernel, iterations=1)
saver.save(dilated, "dilated_combined")

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=20)

line_image = np.copy(image)

# Filter lines based on orientation (vertical lines)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:  # Avoid division by zero
            slope = abs((y2 - y1) / (x2 - x1))
            if slope > 5:  # Vertical lines only
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

saver.save(line_image, "final_detected_lines")

# Display the results
saver.display_images()
