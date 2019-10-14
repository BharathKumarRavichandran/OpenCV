import cv2
import time

# Create video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read current frame
    check, frame = cap.read()
    
    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capturing", gray)

    # Set delay as 1 milli-second
    key = cv2.waitKey(1)
    # Set q as the quit/exit key
    if(key == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()