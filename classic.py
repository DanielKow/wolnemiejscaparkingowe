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

# Apply Gaussian blur to reduce noise before thresholding
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Save the blurred image
save_image_to_run_dir(blurred, "blurred.jpg", run_dir)

# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Save the adaptive threshold image
save_image_to_run_dir(adaptive_thresh, "adaptive_thresh_raw.jpg", run_dir)

# Apply morphological operations to clean noise
kernel = np.ones((3, 3), np.uint8)
morph_cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

# Save the morphologically cleaned image
save_image_to_run_dir(morph_cleaned, "adaptive_thresh_cleaned.jpg", run_dir)

# Optional: Filter small areas using contours
contours, _ = cv2.findContours(morph_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cleaned_contour = np.zeros_like(morph_cleaned)
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Keep only contours with area > 50
        cv2.drawContours(cleaned_contour, [contour], -1, 255, -1)

# Save the contour-filtered result
save_image_to_run_dir(cleaned_contour, "adaptive_thresh_final.jpg", run_dir)

# Display results
images_to_display = [
    gray, blurred, adaptive_thresh, morph_cleaned, cleaned_contour
]
titles = [
    "Grayscale", "Blurred", "Adaptive Threshold (Raw)", "Morph Cleaned", "Final Cleaned"
]

display_images(images_to_display, titles)

print(f"Results saved in: {run_dir}")