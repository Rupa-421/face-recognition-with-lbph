# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:55:04 2021

@author: rupas
"""

import cv2
import numpy as np
from PIL import Image
import os
path='dataset'
recognizer=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    for imagePath in imagePaths:
        print(imagePath)
        PIL_img=Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_img,'uint8')
        id=int(os.path.split(imagePath)[-1].split(".")[0])
        faces=detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print("\n [INFO] Training faces.It will take a few seconds.Wait ..")
ids=[]
faceSamples=[]
faces,ids=getImagesAndLabels(path)
recognizer.train(faces,np.array(ids))
print(ids)
recognizer.write('trainer.yaml')
print("\n [INFO] {0} faces trained .Exiting Program".format(len(np.unique(ids))))
