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

# Perform edge detection
edges = cv2.Canny(blur, 50, 150)
saver.save(edges, "edges_detected")

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=50)

line_image = np.copy(image)

# Filter horizontal lines
if lines is not None:
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the slope
        if x2 - x1 != 0:  # Avoid division by zero
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.1:  # Keep lines that are almost horizontal
                horizontal_lines.append((x1, y1, x2, y2))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

saver.save(line_image, "horizontal_lines_detected")

# Cluster lines into rows (based on y-coordinates)
clustered_lines = np.copy(image)
if horizontal_lines:
    rows = {}
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        row_key = y1 // 20  # Cluster lines based on their approximate vertical position
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append(line)

    # Draw the clustered lines
    for row_key, lines in rows.items():
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(clustered_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

saver.save(clustered_lines, "clustered_lines")

# Display the results
saver.display_images()
