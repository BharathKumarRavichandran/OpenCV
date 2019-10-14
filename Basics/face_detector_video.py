import cv2
import os

# Importing envs
from dotenv import load_dotenv
env_path = '../.env'

# Loading envs
load_dotenv(dotenv_path=env_path)

# Setting globals
HAARCASCADES_DIR = os.getenv('HAARCASCADES_DIR')
FACE_HAARCASCADE_PATH = os.path.join(HAARCASCADES_DIR,'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(FACE_HAARCASCADE_PATH)

# Create video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read current frame
    ret, frame = cap.read()
    
    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face in video feed using face haarcascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face detected, draw rectangle and detect eyes in a face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('Face detection',frame)
    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()