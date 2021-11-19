# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import os
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
id=input("enter a user id => ")
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count=0
while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count+=1
        cv2.imwrite("dataset/"+str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        print(count)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
    cv2.imshow('video',img)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
    elif count==5:
        break
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()