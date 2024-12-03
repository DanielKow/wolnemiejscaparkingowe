import cv2 as cv
from plots import show_images
import numpy as np

def convert_greyscale_to_red(image):
    # Convert grayscale image to RGB format by stacking the same value across 3 channels
    rgb_image = np.stack([image] * 3, axis=-1)

    # Define the lower and upper bounds for the mask
    lower_bound = 180
    upper_bound = 200
    
    # Create the mask using cv.inRange
    mask = cv.inRange(image, lower_bound, upper_bound)

    # Set the pixels in the mask to red ([255, 0, 0])
    rgb_image[mask] = [255, 0, 0]

    return rgb_image



media_directory = 'Media/'
test_directory = 'Test/'
test_image = cv.imread(media_directory + '20241116_151424.jpg', cv.IMREAD_GRAYSCALE)

cv.imwrite(test_directory + 'test.jpg', test_image)

# modified_image = unify_vertical_line_colors(test_image, threshold=50, max_length=6)
# show_images([test_image, modified_image])
# cv.imwrite(test_directory + 'modified.jpg', modified_image)

filtered_image = cv.medianBlur(test_image, ksize=9)
show_images([test_image, filtered_image])
cv.imwrite(test_directory + 'filtered.jpg', filtered_image)

red_image = convert_greyscale_to_red(filtered_image)
show_images([filtered_image, red_image])
cv.imwrite(test_directory + 'red.jpg', red_image)