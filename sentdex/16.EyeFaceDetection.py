# Python script which uses OpenCV and Haarcascades to detect eye and face in a webcam feed.
# Tutorial link: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/

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
HAARCASCADES_DIR = os.getenv('HAARCASCADES_DIR')
FACE_HAARCASCADE_PATH = os.path.join(HAARCASCADES_DIR,'haarcascade_frontalface_default.xml')
EYE_HAARCASCADE_PATH = os.path.join(HAARCASCADES_DIR,'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(FACE_HAARCASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_HAARCASCADE_PATH)

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect face in video feed using face haarcascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face detected, draw rectangle and detect eyes in a face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # Get face region of image
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the ROI using eye haarcascade
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('Eye and Face',frame)
    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()