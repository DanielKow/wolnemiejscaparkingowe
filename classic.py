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

# Create a mask to define areas for stronger blur (e.g., bottom of the image)
height, width, _ = image.shape
mask = np.zeros((height, width), dtype=np.uint8)

# Define the region for stronger blur (e.g., bottom 30% of the image)
cv2.rectangle(mask, (0, int(height * 0.7)), (width, height), 255, -1)  # White rectangle for bottom
saver.save(mask, "blur_mask")

# Apply different levels of blur
light_blur = cv2.GaussianBlur(image, (5, 5), 0)  # Light blur for the whole image
strong_blur = cv2.GaussianBlur(image, (15, 15), 0)  # Strong blur for noisy regions

# Combine the two blurred versions using the mask
blurred_image = cv2.bitwise_and(strong_blur, strong_blur, mask=mask) + \
                cv2.bitwise_and(light_blur, light_blur, mask=cv2.bitwise_not(mask))

saver.save(blurred_image, "selectively_blurred_image")

# Convert to grayscale
gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
saver.save(gray, "grayscale_image")

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
saver.save(enhanced_gray, "enhanced_grayscale")

# Perform edge detection using Canny with optimized thresholds
edges = cv2.Canny(enhanced_gray, 50, 150)
saver.save(edges, "edges_detected")

# Use morphological operations to enhance the edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)  # Optional: Refine edges by erosion
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
