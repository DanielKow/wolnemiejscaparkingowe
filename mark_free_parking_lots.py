import cv2
import numpy as np

image = cv2.imread('empty_marked_full.jpg')

target_color_bgr = [60, 39, 255]

tolerance = 10
lower_bound = np.array([max(color - tolerance, 0) for color in target_color_bgr], dtype=np.uint8)
upper_bound = np.array([min(color + tolerance, 255) for color in target_color_bgr], dtype=np.uint8)

mask = cv2.inRange(image, lower_bound, upper_bound)

cv2.imwrite('mask.jpg', mask)