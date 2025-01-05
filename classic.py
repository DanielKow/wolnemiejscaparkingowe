import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def save_image_to_run_dir(image, filename, run_dir):
    """
    Saves the given image to the specified run directory.
    """
    filepath = os.path.join(run_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Saved: {filepath}")


def display_images(images, titles):
    """
    Displays multiple images using matplotlib.pyplot.
    """
    plt.figure(figsize=(15, 10))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 4, i + 1)
        if len(image.shape) == 2:  # Grayscale
            plt.imshow(image, cmap='gray')
        else:  # Color
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Create a directory to store results
results_path = "results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Determine the next run number
existing_runs = [d for d in os.listdir(results_path) if d.startswith("run_") and d[4:].isdigit()]
next_run_number = 1 + max([int(d[4:]) for d in existing_runs], default=0)

# Create the next run directory
run_dir = os.path.join(results_path, f"run_{next_run_number}")
os.makedirs(run_dir, exist_ok=True)

# Load the original image
image_path = 'test_images/2012-09-11_16_48_36_jpg.rf.4ecc8c87c61680ccc73edc218a2c8d7d.jpg'
image = cv2.imread(image_path)

# Reshape the image to 2D for k-means clustering
pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

# Apply K-means clustering
num_clusters = 8  # Adjust based on the number of color groups in your image
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
labels = kmeans.fit_predict(pixels)
cluster_centers = kmeans.cluster_centers_.astype("uint8")

# Create the clustered image
segmented_image = cluster_centers[labels].reshape(image.shape)

# Save the clustered image
save_image_to_run_dir(segmented_image, "clustered_image.jpg", run_dir)

# Convert the clustered image to grayscale
gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to isolate potential lines
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Save the thresholded image
save_image_to_run_dir(thresholded, "thresholded.jpg", run_dir)

# Edge detection
edges = cv2.Canny(thresholded, 50, 150, apertureSize=3)

# Save the edge-detected image
save_image_to_run_dir(edges, "edges.jpg", run_dir)

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
save_image_to_run_dir(line_image, "detected_lines.jpg", run_dir)

# Display results
images_to_display = [
    image, segmented_image, thresholded, edges, line_image
]
titles = [
    "Original Image", "Clustered Image", "Thresholded", "Edges", "Detected Lines"
]

display_images(images_to_display, titles)

print(f"Results saved in: {run_dir}")
