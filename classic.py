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

# Create a mask for the bottom region
height, width = gray_image.shape
mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(mask, (0, int(height * 0.7)), (width, height), 255, -1)  # Bottom 30%

# Extract the bottom region
bottom_region = cv2.bitwise_and(gray_image, gray_image, mask=mask)
saver.save(bottom_region, "bottom_region_grayscale")

# Reshape the grayscale bottom region for K-means clustering
pixel_values = bottom_region.reshape((-1, 1))
pixel_values = np.float32(pixel_values)  # Convert to float32 for K-means

# Apply K-means clustering
k = 5  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Map pixels to their cluster centers
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(gray_image.shape)

# Combine the segmented bottom region with the original top region
top_region = cv2.bitwise_and(gray_image, gray_image, mask=cv2.bitwise_not(mask))
combined_image = cv2.add(top_region, segmented_image)
saver.save(combined_image, "kmeans_segmented_image")

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_combined = clahe.apply(combined_image)
saver.save(enhanced_combined, "enhanced_combined_image")

# Perform edge detection using Canny
edges = cv2.Canny(enhanced_combined, 50, 150)
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
