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
    raise ValueError("Cannot load the image. Check the file path.")

saver.save(image, "original_image")

# Convert the entire image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
saver.save(gray_image, "grayscale_image")

# Reshape the grayscale image for K-means clustering
pixel_values = gray_image.reshape((-1, 1))
pixel_values = np.float32(pixel_values)  # Convert to float32 for K-means

# Apply K-means clustering
k = 5  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Map pixels to their cluster centers
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(gray_image.shape)
saver.save(segmented_image, "segmented_image")

# Optional: Enhance contrast of the segmented image using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_segmented = clahe.apply(segmented_image)
saver.save(enhanced_segmented, "enhanced_segmented_image")

# Perform edge detection on the enhanced segmented image
edges = cv2.Canny(enhanced_segmented, 50, 150)
saver.save(edges, "edges_detected")

# Use morphological operations to refine edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)
saver.save(eroded, "refined_edges")

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(eroded, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=20)

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
