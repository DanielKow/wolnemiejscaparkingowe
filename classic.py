import cv2
import numpy as np
from saving_results import ResultsSaver

# Initialize ResultsSaver
saver = ResultsSaver()

# Load the image
image_path = '/mnt/data/original_image.jpg'
image = cv2.imread(image_path)
saver.save(image, "original_image")

# Define Region of Interest (ROI)
mask = np.zeros_like(image[:, :, 0])  # Create a single-channel mask
height, width = mask.shape

# Define a polygon that covers the parking lot area
polygon = np.array([
    [0, height * 0.6],  # Start bottom-left, just above the paving
    [width, height * 0.6],  # Bottom-right
    [width, 0],  # Top-right
    [0, 0]  # Top-left
], np.int32)

cv2.fillPoly(mask, [polygon], 255)  # Fill the region with white (255)

# Apply the mask to focus only on the relevant area
masked_image = cv2.bitwise_and(image, image, mask=mask)
saver.save(masked_image, "roi_masked_image")

# Convert to grayscale
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
saver.save(gray, "grayscale_image")

# Apply CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
saver.save(enhanced_gray, "enhanced_grayscale")

# Perform edge detection with adjusted thresholds
edges = cv2.Canny(enhanced_gray, 50, 150)
saver.save(edges, "edges_detected")

# Use morphological operations to enhance edges
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

saver.save(line_image, "vertical_lines_detected_with_roi")

# Display the results
saver.display_images()
