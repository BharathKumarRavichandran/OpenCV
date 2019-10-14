# Python script which uses OpenCV to detect corners using Brute force. The purpose of detecting corners is to track things like motion, do 3D modeling, and recognize objects, shapes, and characters. Homography is also known as Feature Matching.
# Tutorial link: https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/

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
IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'feature-matching-image.jpg')
TEMPLATE_PATH = os.path.join(SAMPLE_IMAGES_DIR,'feature-matching-template.jpg')
OUTPUT_PATH = os.path.join(OUTPUT_DIR,'feature-matching_{}.jpg'.format(__file__))

# Read the image as a color image
img = cv2.imread(IMG_PATH,0)
template = cv2.imread(TEMPLATE_PATH,0)

orb = cv2.ORB_create()

kp_img, des_img           = orb.detectAndCompute(img,None)
kp_template, des_template = orb.detectAndCompute(template,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des_template,des_img)
matches = sorted(matches, key = lambda x:x.distance)

cv2.imshow('Image', img)
cv2.imshow('Template', template)

img_res = cv2.drawMatches(template,kp_template,img,kp_img,matches[:10],None, flags=2)
plt.imshow(img_res)
plt.show()

cv2.waitKey(0) # Wait for any press
cv2.destroyAllWindows() # Destroy all imshow open windows