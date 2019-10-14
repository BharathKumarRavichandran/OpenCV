# Python script which uses OpenCV to read, display and draw on image using matplotlib
# Tutorial link: https://pythonprogramming.net/loading-images-python-opencv-tutorial/

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Importing envs
from dotenv import load_dotenv
env_path = '../.env'

# Loading envs
load_dotenv(dotenv_path=env_path)

# Setting globals
SAMPLE_IMAGES_DIR = os.getenv('SAMPLE_IMAGES_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'four_face.jpg')
OUTPUT_PATH = os.path.join(OUTPUT_DIR,'four_face_{}.jpg'.format(__file__))


img = cv2.imread(IMG_PATH)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.plot([200,300,400],[100,200,300],'c', linewidth=5) # draws a line with width=5
plt.show()

cv2.imwrite(OUTPUT_PATH,img)