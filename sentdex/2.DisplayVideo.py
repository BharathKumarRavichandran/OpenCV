# Python script which uses OpenCV to read webcam feed, display it, convert it to grayscale and write to a file.
# Tutorial link: https://pythonprogramming.net/loading-video-python-opencv-tutorial/

import cv2
import os
import numpy as np

# Importing envs
from dotenv import load_dotenv
env_path = '../.env'

# Loading envs
load_dotenv(dotenv_path=env_path)

# Setting globals
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
OUTPUT_PATH = os.path.join(OUTPUT_DIR,'webcam_{}.avi'.format(__file__))


# Initialise webcam reader
cap = cv2.VideoCapture(0) # 0 to select the first webcam

# Initialise Videowriter to write the video that is read
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH,fourcc, 20.0, (640,480))

# 
while(True):
    # ret gets True/False value based on whether cap is able to read or not
    ret, frame = cap.read()

    # Display original webcam feed
    cv2.imshow('Cam original', frame)

    # Convert feed frames to grayscale and display
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Cam gray',gray)

    # Write video frame to file
    out.write(frame)

    # Listen to keypress and if key is 'q' break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Release the videowriter
out.release()

cv2.destroyAllWindows() # Destroy all imshow open windows