# Python script which uses OpenCV to show different types of simple thresholding in OpenCV.
# Tutorial link: https://pythonprogramming.net/thresholding-image-analysis-python-opencv-tutorial/

import cv2
import os
import numpy as np


# Importing envs
from dotenv import load_dotenv
env_path = '../.env'

# Loading envs
load_dotenv(dotenv_path=env_path)

# Setting globals
SAMPLE_IMAGES_DIR = os.getenv('SAMPLE_IMAGES_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'bookpage.jpg')
OUTPUT_PATH = os.path.join(OUTPUT_DIR,'bookpage_{}.jpg'.format(__file__))

# Read the image as a color image
img = cv2.imread(IMG_PATH)
# Converting the color image to grayscale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Thresholding a color image
color_ret, color_threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

# Thresholding the grayscale image
grayscale_ret, grayscale_threshold = cv2.threshold(grayscaled_img, 10, 255, cv2.THRESH_BINARY)

# Thresholding the grayscale image using gaussian threshold which gives clear distinction in the book pages
gauss_threshold = cv2.adaptiveThreshold(grayscaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

# Thresholding the grayscale using OTSU threshold
otsu_ret, otsu_threshold = cv2.threshold(grayscaled_img, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('Original Image', img)
cv2.imshow('Grayscaled Image', grayscaled_img)
cv2.imshow('Color Threshold', color_threshold)
cv2.imshow('Grayscale Threshold', grayscale_threshold)
cv2.imshow('Gaussian Threshold', gauss_threshold)
cv2.imshow('OTSU Threshold', otsu_threshold)

cv2.waitKey(0) # Wait for any press
cv2.destroyAllWindows() # Destroy all imshow open windows