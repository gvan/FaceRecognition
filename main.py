import cv2
import matplotlib.pyplot as plt
import numpy as np

print("hello")
imagePath = 'people.jpeg'

img = cv2.imread(imagePath)

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faceClasifier = cv2.CascadeClassifier(
  cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = faceClasifier.detectMultiScale(
  grayImage, scaleFactor = 1.1, minNeighbors = 5, minSize=(40,40)
)

for(x, y, w, h) in face:
  croppedImg = img[y:y+h, x:x+w]
  cv2.imwrite("image%d.jpeg"%(w), croppedImg)

for(x, y, w, h) in face:
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

cv2.imwrite("newImage.jpeg", img)