# Python script which uses OpenCV to recognise objects using template matching method.
# Tutorial link: https://pythonprogramming.net/template-matching-python-opencv-tutorial/

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
TEMPLATE_MAIN_IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'template_main.jpg')
TEMPLATE_PORT_IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'template_port.jpg')

OUTPUT_PATH = os.path.join(OUTPUT_DIR,'template_matching_{}.jpg'.format(__file__))

# Read the image
img_rgb = cv2.imread(TEMPLATE_MAIN_IMG_PATH)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Copy the image to draw circles
img_detected = img_rgb.copy()

# Read template image (i.e. port)
template = cv2.imread(TEMPLATE_PORT_IMG_PATH,0)
w, h = template.shape[::-1]

# Match template with grayscale version of original image
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.9 # Keeping 80% match as the threshold
loc = np.where( res >= threshold)

# Draw yellow rectangles around matched parts
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_detected, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Main image', img_rgb)
cv2.imshow('Object to detect', template)
cv2.imshow('Detected', img_detected)

# Write image to a file
cv2.imwrite(OUTPUT_PATH, img_rgb)

cv2.waitKey(0) # Wait for any press
cv2.destroyAllWindows() # Destroy all imshow open windows