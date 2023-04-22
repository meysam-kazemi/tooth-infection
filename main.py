import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import pydicom as dicom

model = torch.hub.load('ultralytics/yolov5','custom',
       path='weights/best.pt',force_reload=False)

res = model("1.2.246.512.1.2.0.4.3123315755482.18975174176.20221231095825.png")
crp = res.crop(save=False)

length = len(crp)
crps = np.zeros((length,60,60))
for i in range(length):
    im = cv2.cvtColor(crp[i]["im"],cv2.COLOR_BGR2GRAY)
    crps[i] = cv2.resize(im,(60,60))
    
crps = crps.reshape(crps.shape+(1,))
network = tf.keras.models.load_model("first_network.h5")

def predict(x):
    pred = network.predict(x)
    pred[pred[:,1]<0.7] = 0
    pred = np.argmax(pred,axis=1)
    return 1 if np.sum(pred)>0 else 0

p = predict(crps)
print(p)


