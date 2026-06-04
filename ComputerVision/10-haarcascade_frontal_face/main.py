# Haarscascade Frontal Face

import cv2
import os
import numpy as np

train_path = './dataset/train/'

face_list = []
class_list = []

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

person_name = os.listdir(train_path)
print(person_name)

for idx, name in enumerate(person_name):
    full_path = train_path + '/' + name
    
    for img_name in os.listdir(full_path):
        img_full_path = full_path + '/' + img_name
        img = cv2.imread(img_full_path, 0) # grey image
        
        detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
        
        if len(detected_face) < 1:
            continue
        
        for face_rent in detected_face:
            x, y, h, w = face_rent
            face_img = img[y:y+h, x:x+w]
            
            face_list.append(face_img)
            class_list.append(idx)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

test_path = 'dataset/test'

for img_name in os.listdir(test_path):
    full_img_path = test_path + '/' + img_name
    img_gray = cv2.imread(full_img_path, 0)
    img_bgr = cv2.imread(full_img_path)
    
    detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
    
    if len(detected_face) < 1:
        continue
    
    for face_rent in detected_face:
        x, y, h, w = face_rent
        face_img = img_gray[y:y+h, x:x+w]
        
        res, distance = face_recognizer.predict(face_img)
        
        # distance -> seberapa tidak mirip, makin gede = tidak mirip
        # confidence -> seberapa yakin terhadap hasil prediksi
        
        if distance < 100:
            confidence_percent = round(100 - distance)
        else:
            confidence_percent = 0
            
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        text = f"{person_name[res]} : {confidence_percent}%"
        cv2.putText(img_bgr, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0,255,0), 2)
        
        cv2.imshow('Result', img_bgr)
        cv2.waitKey(0)