import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import pydicom as dicom
from time import time
import argparse


# Load models
model = torch.hub.load('yolov5','custom', # Yolo pre-trained model (for crop dentals)
       path='weights/best.pt',source="local")
network = tf.keras.models.load_model("first_network.h5",compile=False) # Neural network model
network.compile(loss="binary_crossentropy",
    optimizer="adam",
    metrics=["acc"]
             )
def predict(x): 
    pred = network.predict(x,verbose=0)
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
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",help="path to folder containig test images.")
    parser.add_argument("--output", help="path to final csv output")
    args = parser.parse_args()
    
    # Images directory
    images_path = os.listdir(args.inputs)
    images_length = len(images_path)
    images = np.zeros((images_length,300,600),dtype="uint8") # An empty arrat for saving images
    sop = []
    for i,im_path in enumerate(images_path):
        im = dicom.dcmread(args.inputs+"/"+im_path) # Read dicom image
        sop.append(im.SOPInstanceUID )# Save SopInstanceUID in a variable(for using in csv file)
        im = ((im.pixel_array/65535)*255).astype('uint8') # Convert dtype of images to uint8
        im = cv2.resize(im,(600,300)) # Resize image for yolo(300*600)| cv2  is inverse !!!
        images[i] = im
        
    
    labels = apply_model(images) # Predict labels of test images
    df = pd.DataFrame({"SOPInstanceUID":sop,"Labels":labels}) # Create DataFrame of labels
    print("="*10+" csv "+"="*10)
    print(df.head(10),"\n"+"="*30)
    # Save DataFrame
    if args.output.endswith(".csv"):    
        df.to_csv(args.output)
    else:
        df.to_csv(args.output+".csv")
    print("CSV file saved\u2713")



