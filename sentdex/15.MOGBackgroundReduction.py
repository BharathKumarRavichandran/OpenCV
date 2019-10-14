# Python script which uses OpenCV to reduce the background of images, by detecting motion.
# Tutorial link: https://pythonprogramming.net/mog-background-reduction-python-opencv-tutorial/

import cv2
import os
import numpy as np

# Importing envs
from dotenv import load_dotenv
env_path = '../.env'

# Loading envs
load_dotenv(dotenv_path=env_path)

# Setting globals
SAMPLE_VIDEOS_DIR = os.getenv('SAMPLE_VIDEOS_DIR')
VIDEO_PATH = os.path.join(SAMPLE_VIDEOS_DIR,'people-walking.mp4')

# Put 0 as parameter to use webcam feed.
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(VIDEO_PATH)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()