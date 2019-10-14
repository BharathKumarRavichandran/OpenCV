import cv2
import datetime
import pandas as pd
import time

first_frame = None
status_list = [None, None]
times = []
df = pd.DataFrame(columns=["Start", "End"])

# Create video capture object
video = cv2.VideoCapture(0)

a = 1

while True:
    a = a+1

    # Read current frame
    check, frame = video.read()
    status = 0
    
    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert gray scale frame to Gaussian Blur frame
    gauss = cv2.GaussianBlur(gray, (21,21), 0)

    # Store first_frame of the video
    if(first_frame is None):
        first_frame = gauss
        continue

    # Calculate the difference between the first frame and the other frames
    delta_frame = cv2.absdiff(first_frame, gauss)
    
    # Provide a threshold value, such that it will convert the difference value with less than 30 to black. If the difference is greater than 30 it will convert those pixels to white
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)

    # Find contours to objects that move
    (cnts,_) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # Remove noises and shadows
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1

        # Add rectangular box around the moving object in the frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    
    status_list.append(status)
    status_list = status_list[-2:]

    # Add Start and End time
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.datetime.now())

    # Draw/Show all frames
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('delta', delta_frame)
    cv2.imshow('thresh', thresh_delta)

    # Set delay as 1 milli-second
    key = cv2.waitKey(1)
    # Set q as the quit/exit key
    if key == ord('q'):
        break

# Store time values in a dataframe
for i in range(0, len(times), 2):
    df =df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)
# Convert the time values to csv
df.to_csv("out/times.csv")

video.release()
cv2.destroyAllWindows()