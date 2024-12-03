import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def show_images(images):
    for number, img in enumerate(images):
        plt.subplot(1, len(images), number+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

# Load the image
media_directory = 'Media_heic/'
image = cv.imread(media_directory + '20241110_164917.heic', cv.IMREAD_GRAYSCALE)

# Apply the Canny edge detection
edges = cv.Canny(image, 100, 200)

# Display the images
show_images([image, edges])