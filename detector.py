import numpy as np
import cv2

input = cv2.imread('img.jpeg')
gray_img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)
for (x,y,w,h) in faces:
    output = cv2.rectangle(input, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow("output",output)

cv2.waitKey(0)
cv2.destroyAllWindows()