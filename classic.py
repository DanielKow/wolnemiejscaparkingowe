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

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)
saver.save(blur, "blurred_image")

# Perform edge detection with adjusted thresholds
edges = cv2.Canny(blur, 30, 100)  # Adjusted thresholds for better edge detection
saver.save(edges, "edges_detected")

# Use Hough Line Transform to detect lines with stricter criteria
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=20)
line_image = np.copy(image)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

saver.save(line_image, "lines_detected")

# Additional step: Highlight parallel lines and refine parking line detection
parking_lines = np.copy(image)

if lines is not None:
    slopes = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:  # Avoid division by zero
            slope = (y2 - y1) / (x2 - x1)
            slopes.append((slope, line[0]))

    # Sort lines by slope and identify potential parking slots
    slopes.sort(key=lambda x: x[0])
    for i in range(len(slopes) - 1):
        slope1, line1 = slopes[i]
        slope2, line2 = slopes[i + 1]
        if abs(slope1 - slope2) < 0.05:  # Parallel slope threshold
            # Draw parallel lines
            x1, y1, x2, y2 = line1
            cv2.line(parking_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
            x1, y1, x2, y2 = line2
            cv2.line(parking_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

saver.save(parking_lines, "parking_lines_refined")

# Display the results
saver.display_images()
