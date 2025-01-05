import cv2
import numpy as np
from PIL import ImageEnhance

from sklearn.cluster import KMeans
from saving_results import ResultsSaver

# Initialize the results saver
results_saver = ResultsSaver()


# Load the original image
image_path = 'test_images/2012-09-11_16_48_36_jpg.rf.4ecc8c87c61680ccc73edc218a2c8d7d.jpg'
image = cv2.imread(image_path)


# Reshape the image to 2D for k-means clustering
pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

# Apply K-means clustering
num_clusters = 32  # Adjust based on the number of color groups in your image
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
labels = kmeans.fit_predict(pixels)
cluster_centers = kmeans.cluster_centers_.astype("uint8")

# Create the clustered image
segmented_image = cluster_centers[labels].reshape(image.shape)

# Save the clustered image
results_saver.save(segmented_image, "clustered_image")

# Convert the clustered image to grayscale
gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to isolate potential lines
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Save the thresholded image
results_saver.save(thresholded, "thresholded")

# Edge detection
edges = cv2.Canny(thresholded, 50, 150, apertureSize=3)

# Save the edge-detected image
results_saver.save(edges, "edges")

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20)

# Draw the detected lines on the original image
line_image = np.copy(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Filter lines that are approximately parallel
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        if 85 <= angle <= 95 or -5 <= angle <= 5:  # Near-vertical or near-horizontal
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the final result image
results_saver.save(line_image, "detected_lines")

results_saver.display_images()
