# Python script which uses OpenCV to detect corners. The purpose of detecting corners is to track things like motion, do 3D modeling, and recognize objects, shapes, and characters.
# Tutorial link: https://pythonprogramming.net/corner-detection-python-opencv-tutorial/

import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# Importing envs
from dotenv import load_dotenv
env_path = '../.env'

# Loading envs
load_dotenv(dotenv_path=env_path)

# Setting globals
SAMPLE_IMAGES_DIR = os.getenv('SAMPLE_IMAGES_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'corner-detection.jpg')
OUTPUT_PATH = os.path.join(OUTPUT_DIR,'corner-detection_{}.jpg'.format(__file__))

# Read the image as a color image
img = cv2.imread(IMG_PATH)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Parameters: (image, max corners to detect, quality, minimum distance between corners)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)
    
cv2.imshow('Corners',img)

cv2.waitKey(0) # Wait for any press
cv2.destroyAllWindows() # Destroy all imshow open windows