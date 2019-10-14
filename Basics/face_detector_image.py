import cv2
import os

# Importing envs
from dotenv import load_dotenv
env_path = '../.env'

# Loading envs
load_dotenv(dotenv_path=env_path)

# Setting globals
SAMPLE_IMAGES_DIR = os.getenv('SAMPLE_IMAGES_DIR')
IMG_PATH = os.path.join(SAMPLE_IMAGES_DIR,'four_face.jpg')

HAARCASCADES_DIR = os.getenv('HAARCASCADES_DIR')
FACE_HAARCASCADE_PATH = os.path.join(HAARCASCADES_DIR,'haarcascade_frontalface_alt.xml')

# Create a CascadeClassifier Object
# XML file contains the face features
face_cascade = cv2.CascadeClassifier(FACE_HAARCASCADE_PATH)

# Reading the image as it is
img = cv2.imread(IMG_PATH)

# Reading the image as gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Search the co-ordinates of face in the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

# Create rectangle around detected face
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 5)

resized_img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

cv2.imshow("Gray", resized_img)
cv2.waitKey(0)

cv2.destroyAllWindows()