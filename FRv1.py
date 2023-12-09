import cv2
print(cv2.__version__)

cas=cv2.CascadeClassifier("C:/Users/khush/OneDrive/Desktop/project exhibhition 2/1.XML")
img=cv2.imread('WhatsApp Image 2023-01-16 at 7.31.41 PM')
cv2.imshow('cara',img)
cv2.waitKey(0)
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('cara',gray)
#cv2.waitKey(0)

#fac=cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8)

#for (x,y,w,h) in fac:
#    cv2.rectangle(img,(x,y),(x+w,y+h),(250,0,0),3)

#cv2.imshow('Output',img)
#cv2.waitKey(0)
