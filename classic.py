import cv2
import numpy as np
from saving_results import ResultsSaver

# Initialize ResultsSaver
saver = ResultsSaver()

# Load the image
image_path = 'test_images/2012-09-11_16_48_36_jpg.rf.4ecc8c87c61680ccc73edc218a2c8d7d.jpg'
image = cv2.imread(image_path)
saver.save(image, "original_image")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
saver.save(gray, "grayscale_image")

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
saver.save(enhanced_gray, "enhanced_grayscale")

# Perform edge detection
edges = cv2.Canny(enhanced_gray, 50, 150)
saver.save(edges, "edges_detected")

# Use morphological operations to enhance the edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)
saver.save(dilated, "dilated_edges")

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

line_image = np.copy(image)

# Filter vertical lines
if lines is not None:
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:  # Avoid division by zero
            slope = abs((y2 - y1) / (x2 - x1))
            if slope > 5:  # Keep lines that are almost vertical
                vertical_lines.append((x1, y1, x2, y2))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

saver.save(line_image, "vertical_lines_detected")

# Display the results
saver.display_images()
