import cv2
import numpy as np

image = cv2.imread('empty_marked_full.jpg')

target_color_bgr = [60, 39, 255]

tolerance = 10
lower_bound = np.array([max(c - tolerance, 0) for c in target_color_bgr], dtype=np.uint8)
upper_bound = np.array([min(c + tolerance, 255) for c in target_color_bgr], dtype=np.uint8)

mask = cv2.inRange(image, lower_bound, upper_bound)

cv2.imwrite('mask.jpg', mask)