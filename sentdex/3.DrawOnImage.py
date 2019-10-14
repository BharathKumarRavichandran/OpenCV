# Python script which uses OpenCV to draw line, rectangle, circle and polygon over image.
# Tutorial link: https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/

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
IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'deadpool.jpg')
OUTPUT_PATH = os.path.join(OUTPUT_DIR,'deadpool_{}.jpg'.format(__file__))

# Read the image as a color image
img = cv2.imread(IMG_PATH,cv2.IMREAD_COLOR)

# Draw line
cv2.line(img,(0,0),(200,300),(255,255,255),50)

# Draw rectangle
cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15)

# Draw circle
cv2.circle(img,(447,63), 63, (0,255,0), -1)

# Draw polygon
# Feed points as an array using numpy array
pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
# Reshape the array to a 1 x 2
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3) # True connects first and last point

# Write text over an image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,' Bharath Kumar',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)
cv2.imshow('image',img)

# Write image to a file
cv2.imwrite(OUTPUT_PATH,img)

cv2.waitKey(0) # Wait for any press
cv2.destroyAllWindows() # Destroy all imshow open windows