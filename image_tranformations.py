import numpy as np
import cv2 as cv

def white_to_red(image):
    # Split the image into RGB channels
    red, green, blue = cv.split(image)

    # Create a new image with the red channel set to 255
    red_image = np.zeros_like(image)
    red_image[:, :, 0] = 255

    # Combine the red channel with the original green and blue channels
    new_image = cv.merge((red_image, green, blue))

    return new_image

def unify_vertical_line_colors(image, threshold=50, max_length=10):
    # Get image dimensions
    height, width = image.shape

    # Create a copy of the image to hold the modifications
    unified_image = image.copy()

    # Iterate over each column
    for x in range(width):
        y = 0
        while y < height - 1:
            # Start of potential unified segment
            start_y = y
            count = 0  # to keep track of the segment length

            # Continue along the column while colors are within threshold and max length is not exceeded
            while y < height - 1 and abs(int(image[y, x]) - int(image[y + 1, x])) <= threshold:
                y += 1
                count += 1
                if count >= max_length:
                    break

            # End of the unified segment
            end_y = y

            # If the segment is valid (matching within the threshold and > 1 in length)
            if start_y != end_y:
                avg_color_value = int(np.mean(image[start_y:end_y + 1, x]))

                # Set these pixels to the average value
                unified_image[start_y:end_y + 1, x] = avg_color_value

            # Move to the next pixel to start a new potential segment
            y += 1

    return unified_image