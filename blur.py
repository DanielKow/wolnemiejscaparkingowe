import cv2 as cv

def apply_gaussian_blur(image):
    return cv.GaussianBlur(image, (15, 15), sigmaX=100)