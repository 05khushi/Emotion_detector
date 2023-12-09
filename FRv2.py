import os
import cv2
import numpy as np

img_features = []
img_lable = []
cas = cv2.CascadeClassifier("D:/Coding/Python/1.XML")

def create_train_data():


    people = ['angry','disgust','fear', 'happy', 'neutral','sad','surprise']

    directory = r"D:\Coding\Python\Face_img"




    for person in people:
        path_for_person=os.path.join(directory,person)
        img_lables=people.index(person)

        for img in os.listdir(path_for_person):
            img_path= os.path.join(path_for_person,img)

            img_array=cv2.imread(img_path)
            gray_image=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

            faces_rect= cas.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=3)

            for (x,y,w,h) in faces_rect:
                faces_roi=gray_image[y:y+h,x:x+w]
                img_features.append(faces_roi)
                img_lable.append(img_lables)

create_train_data()

img_features=np.array(img_features,dtype='object')
img_lable=np.array(img_lable)

print(f'{len(img_features)}')
print(f'{len(img_lable)}')


face_recogniszer=cv2.face.LBPHFaceRecognizer_create()

face_recogniszer.train(img_features,img_lable)

face_recogniszer.save('face_trained.yml')

np.save('features.npy',img_features)
np.save('lables.npy',img_lable)

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    people = ['angry','disgust','fear', 'happy', 'neutral','sad','surprise']

    features = np.load('features.npy',allow_pickle=True)
    lables = np.load('lables.npy')

    face_recogniszer.read('face_trained.yml')

    success,img=cap.read()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fac_rect=cas.detectMultiScale(imggray,scaleFactor=1.1,minNeighbors=3)

    for (x,y,w,h) in fac_rect:
        cv2.rectangle(img,(x,y),(x+w,y+h),(250,0,0),3)

    for (x,y,w,h) in fac_rect:
        faces_roi=imggray[y:y+h,x:x+w]
        img_lable,confidence=face_recogniszer.predict(faces_roi)
        cv2.putText(img,str(people[img_lable]),(20,20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
