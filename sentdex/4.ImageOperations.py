# Python script which uses OpenCV to show different image operations.
# Tutorial link: https://pythonprogramming.net/image-operations-python-opencv-tutorial/

import cv2
import os

# Importing envs
from dotenv import load_dotenv
env_path = '../.env'

# Loading envs
load_dotenv(dotenv_path=env_path)

# Setting globals
SAMPLE_IMAGES_DIR = os.getenv('SAMPLE_IMAGES_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'deadpool.jpg')
OUTPUT_PATH = os.path.join(OUTPUT_DIR,'deadpool_{}.jpg'.format(__file__))

# Read the image as a color image
img = cv2.imread(IMG_PATH,cv2.IMREAD_COLOR)

# Reference specific pixel
px = img[55,55]

# Change a pixel value
img[55,55] = [255,255,255]
cv2.imshow('Pixel change',img)

# Re-reference
px = img[55,55]
print('Re-reference pixel: {}'.format(px))

# Reference a ROI, or Region of Image
px = img[100:150,100:150]
print('Reference a Region of Image(ROI): \n{}'.format(px))

# Modify ROI
img[100:150,100:150] = [255,255,255]
cv2.imshow('Modified ROI',img)

# Reference certain characteristics of our image
print('Image shape: {}'.format(img.shape))
print('Image size: {}'.format(img.size))
print('Image datatype: {}'.format(img.dtype))

# 
roi_crop = img[37:111,107:194]
img[0:74,0:87] = roi_crop
cv2.imshow('ROI overlay',img)

# Write image to a file
# cv2.imwrite(OUTPUT_PATH,img)

cv2.waitKey(0) # Wait for any press
cv2.destroyAllWindows() # Destroy all imshow open windows