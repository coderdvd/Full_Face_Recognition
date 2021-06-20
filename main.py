import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray_frame, 1.1, 5, None, (30, 30))
    eyes = eyeCascade.detectMultiScale(gray_frame, 1.1, 5, None, (30, 30))
    smiles = smileCascade.detectMultiScale(gray_frame, 1.1, 5, None, (30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2) # Blue Green Red

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2) # Blue Green Red

    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2) # Blue Green Red

    cv2.imshow('Full Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
