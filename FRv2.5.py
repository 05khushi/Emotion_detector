import os
import cv2
import numpy as np

cas = cv2.CascadeClassifier("D:/Coding/Python/1.XML")

people = ['angry','disgust','fear', 'happy', 'neutral','sad','surprise']

features = np.load('features.npy',allow_pickle=True)
lables = np.load('lables.npy')
face_recogniszer=cv2.face.LBPHFaceRecognizer_create()
face_recogniszer.read('face_trained.yml')

img=cv2.imread('IMG_20221019_175625 (2).jpg')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fac_rect=cas.detectMultiScale(imggray,scaleFactor=1.1,minNeighbors=3)

for (x,y,w,h) in fac_rect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(250,0,0),3)

for (x,y,w,h) in fac_rect:
    faces_roi=imggray[y:y+h,x:x+w]
    img_lable,confidence=face_recogniszer.predict(faces_roi)
    cv2.putText(img,str(people[img_lable]),(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),thickness=2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv2.imshow('vedio', img)
cv2.waitKey(0)