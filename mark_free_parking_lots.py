import cv2
import numpy as np

# Load the image
image = cv2.imread('empty_marked_full.jpg')
print(image[475, 120])
# Define the target color (BGR instead of RGB)
target_color_bgr = [60, 39, 255]  # Convert RGB(254, 40, 60) to BGR(60, 40, 254)

tolerance = 10
lower_bound = np.array([max(c - tolerance, 0) for c in target_color_bgr], dtype=np.uint8)
upper_bound = np.array([min(c + tolerance, 255) for c in target_color_bgr], dtype=np.uint8)


# Create the mask
mask = cv2.inRange(image, lower_bound, upper_bound)

# Save or display the mask
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
