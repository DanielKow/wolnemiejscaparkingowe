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

# Create a mask for the bottom region
height, width, _ = image.shape
mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(mask, (0, int(height * 0.7)), (width, height), 255, -1)  # Bottom 30%

# Extract the bottom region for K-means clustering
bottom_region = cv2.bitwise_and(image, image, mask=mask)

# Reshape the bottom region for K-means clustering
pixel_values = bottom_region.reshape((-1, 3))
pixel_values = np.float32(pixel_values)  # Convert to float32 for K-means

# Apply K-means clustering
k = 4  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Map pixels to their cluster centers
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(bottom_region.shape)

# Combine the clustered bottom region with the original top region
top_region = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
combined_image = cv2.addWeighted(top_region, 1, segmented_image, 1, 0)
saver.save(combined_image, "kmeans_segmented_image")

# Convert to grayscale
gray = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)
saver.save(gray, "grayscale_image")

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
saver.save(enhanced_gray, "enhanced_grayscale")

# Perform edge detection using Canny with optimized thresholds
edges = cv2.Canny(enhanced_gray, 50, 150)
saver.save(edges, "edges_detected")

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=20)

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
