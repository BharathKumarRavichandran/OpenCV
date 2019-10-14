# Python script which uses OpenCV to detect foreground from background.
# Tutorial link: https://pythonprogramming.net/grabcut-foreground-extraction-python-opencv-tutorial/

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
IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'grabcut_sentdex.jpg')
OUTPUT_PATH = os.path.join(OUTPUT_DIR,'grabcut_sentdex_{}.jpg'.format(__file__))

# Read the image as a color image
img = cv2.imread(IMG_PATH,cv2.IMREAD_COLOR)
mask = np.zeros(img.shape[:2],np.uint8)

# Define the bg and fg models
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# Define the face object rectangle
rect = (161,79,150,150)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# Blacken parts where mask is 0 or 2
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()