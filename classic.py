import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


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

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise slightly
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Save the blurred image
save_image_to_run_dir(blurred, "blurred.jpg", run_dir)

# Apply adaptive thresholding with tuned parameters
adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
)

# Save the adaptive threshold image
save_image_to_run_dir(adaptive_thresh, "adaptive_thresh_tuned.jpg", run_dir)

# OPTIONAL: Morphological operations (with a smaller kernel)
kernel = np.ones((2, 2), np.uint8)  # Very small kernel
morph_cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

# Save the morphologically cleaned image
save_image_to_run_dir(morph_cleaned, "adaptive_thresh_morph_cleaned.jpg", run_dir)

# Edge detection
edges = cv2.Canny(morph_cleaned, 50, 150, apertureSize=3)

# Save the edge-detected image
save_image_to_run_dir(edges, "edges.jpg", run_dir)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

# Draw the detected lines on the original image
line_image = np.copy(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Filter lines based on their orientation (e.g., mostly horizontal or vertical)
        if abs(y2 - y1) < 20 or abs(x2 - x1) < 20:  # Adjust this threshold as needed
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the final result image
save_image_to_run_dir(line_image, "detected_lines.jpg", run_dir)

# Display results
images_to_display = [
    gray, blurred, adaptive_thresh, morph_cleaned, edges, line_image
]
titles = [
    "Grayscale", "Blurred", "Adaptive Threshold (Tuned)", "Morph Cleaned", "Edges", "Detected Lines"
]

display_images(images_to_display, titles)

print(f"Results saved in: {run_dir}")
