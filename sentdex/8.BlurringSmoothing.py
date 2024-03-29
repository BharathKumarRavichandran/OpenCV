# Python script which uses OpenCV to eliminate noises using simple thresholds.
# Tutorial link: https://pythonprogramming.net/blurring-smoothing-python-opencv-tutorial/

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel = np.ones((15,15),np.float32)/225
    smoothed  = cv2.filter2D(res,-1,kernel)
    blur      = cv2.GaussianBlur(res,(15,15),0)
    median    = cv2.medianBlur(res,15)
    bilateral = cv2.bilateralFilter(res,15,75,75)
    
    cv2.imshow('Original',frame)
    cv2.imshow('Averaging',smoothed)
    cv2.imshow('Gaussian Blurring',blur)
    cv2.imshow('Median Blur',median)
    cv2.imshow('Bilateral Blur',bilateral)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()