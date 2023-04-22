import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import pydicom as dicom
from time import time

path = "/run/media/meysam/PROGRAM/0.data/iaaa/images"
images = os.listdir(path)
images = list(map(lambda a:path+"/"+a,images))

model = torch.hub.load('yolov5','custom',
       path='weights/best.pt',source="local")
network = tf.keras.models.load_model("first_network.h5")

def predict(x):
    pred = network.predict(x)
    pred[pred[:,1]<0.7] = 0
    pred = np.argmax(pred,axis=1)
    return 1 if np.sum(pred)>0 else 0


def apply_model(images):
    pred = np.empty((len(images),))
    for i,im in enumerate(images):
        res = model(im)
        crp = res.crop(save=False)
        
        length = len(crp)
        crps = np.zeros((length,60,60))
        for i in range(length):
            im = cv2.cvtColor(crp[i]["im"],cv2.COLOR_BGR2GRAY)
            crps[i] = cv2.resize(im,(60,60))
            
        crps = crps.reshape(crps.shape+(1,))
        
        pred[i] = predict(crps)
    return pred
        
s = time()
p = apply_model(images[:100])
e = time()
print("="*20+"\ntime of run :",round(e-s,2))