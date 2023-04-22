import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import pydicom as dicom
from time import time


# Path of dicom images
path = "/run/media/meysam/PROGRAM/0.data/iaaa/images"
images_path = os.listdir(path)
images_path = list(map(lambda a:path+"/"+a,images))

images_length = len(images) # Number of images
images = np.zeros((images_length,300,600),dtype="uint8") # Empty array for images
sop = np.zeros(images_length,dtype="str")
for i,im_path in enumerate(images_path):
    im = dicom.dcmread(im_path) # Read dicom image
    sop[i] = im.SOPInstanceUID # Save SopInstanceUID in a variable(for using in csv file)
    im = ((im.pixel_array/65535)*255).astype('uint8') # Convert dtype of images to uint8
    im = cv2.resize(im,(600,300)) # Resize image for yolo(300*600)| cv2  is inverse !!!
    images[i] = im
    
model = torch.hub.load('yolov5','custom', # Yolo pretrained model (for crop dentals)
       path='weights/best.pt',source="local")
network = tf.keras.models.load_model("first_network.h5") # Neural network model

def predict(x): 
    pred = network.predict(x)
    # If the probability of the photo being abnormal is less than 70%, we assume it is normal.
    pred[pred[:,1]<0.7] = 0 
    pred = np.argmax(pred,axis=1)
    return 1 if np.sum(pred)>0 else 0 # normals=>0 | abnormals=>1

def apply_model(images):
    pred = np.empty((len(images),),dtype="uint8") # Save labels that predict model in a variable(for using in csv file)
    for j,im in enumerate(images):
        res = model(im)
        crp = res.crop(save=False)
        
        length = len(crp)
        crps = np.zeros((length,60,60))
        
        for i in range(length):
            im = cv2.cvtColor(crp[i]["im"],cv2.COLOR_BGR2GRAY)
            crps[i] = cv2.resize(im,(60,60))
            
        crps = crps.reshape(crps.shape+(1,))
        
        pred[j] = predict(crps)
    return pred
        
s = time()
labels = apply_model(images)
e = time()
print("="*20+"\ntime of run :",round(e-s,2))

df = pd.Dataframe({"SOPInstanceUID":sop,"Labels":labels)
print(df.head(5))



