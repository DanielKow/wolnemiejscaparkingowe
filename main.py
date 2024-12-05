import cv2 as cv
import os
import glob
import numpy as np
from plots import show_images

def get_jpg_files_without_extension(directory):
    jpg_files = glob.glob(os.path.join(directory, "*.jpg"))
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in jpg_files]
    return base_names

def draw_rectangles_around_cars(image, edges):
    color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if 500 < cv.contourArea(contour) < 5000:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    return color_image


def save_test_images(image, name, suffix):
    cv.imwrite(f"test/{name}-{suffix}.jpg", image)

def find_parking_slots(image_name):
    image = cv.imread(media_directory + image_name + '.jpg', cv.IMREAD_GRAYSCALE)
    save_test_images(image, image_name, 'test')
    
    blurred = cv.medianBlur(image, 1)
    save_test_images(blurred, image_name, 'blurred')

    dilated = cv.dilate(blurred,np.ones((3,3)))
    save_test_images(dilated, image_name, 'dilated')

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    closing = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
    
    edges = cv.Canny(closing, threshold1=50, threshold2=150)
    save_test_images(edges, image_name, 'edges')

    rectangles = draw_rectangles_around_cars(image, edges)
    save_test_images(rectangles, image_name, 'rectangles')

media_directory = 'Media/'

jpg_files = ['1', '2', '3', '4', '5', '6', '7']

for image_name in jpg_files:
    find_parking_slots(image_name)
