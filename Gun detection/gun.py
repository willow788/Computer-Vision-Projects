import cv2
import numpy as np
import imutils
import time
import datetime

gun_cascade = cv2.CascadeClassifier('cascade.xml')

if gun_cascade.empty():
    raise RuntimeError("Failed to load cascade.xml")

#initialising the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

#initialisng the first frame and gun_exist variable
first_frame = None
gun_exist = False

while True:
    ret, frame = camera.read()
    gun_exist = False

    if not ret or frame is None:
        print("Can't receive frame. Exiting ...")
        break
    frame = imutils.resize(frame, width=500)
    grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecting the gun
    gun = gun_cascade.detectMultiScale(grayed, 1.3, 20, minSize=(100, 100))
    if len(gun) > 0:
        gun_exist = True
    for (x, y, w, h) in gun:
        
        #drawing a frame around the gun
        frame = cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

        #region of interest in grayed and color frame
        roi_gray = grayed[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    if first_frame is None:
        first_frame = grayed
        continue
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    if gun_exist is True:
        cv2.putText(frame, "Gun Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Gun Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Son of a Gun", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
