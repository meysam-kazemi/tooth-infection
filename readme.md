# IAAA

## The purpose of this project
- In this competition, we have to predict whether these images have damaged (infected) teeth or not using DICOM data.
## Steps
- In this test, I first used the YOLO5 to examine each tooth separately.I labeled about 200 images to train Yolo.
## Problems
- One of the problems of this contest is the lack of abnormal photos.
- The number of abnormal teeth is 71, and the number of normal teeth is about 8000, and this reduces the performance of the model.
## Solution
- One of the methods that I thought would solve this problem to some extent was augmenting abnormal images.


# TODO:
-[x] Read DICOM images and save them in png format(to train Yolo).

-[x] Labeling some of the images for Yolo.

-[x] Train Yolo5 and save the model.

-[x] Separating normal images from abnormals.

-[x] Crop normal images with Pre-trained Yolo and save them in a new folder.

-[x] Crop abnormal images using their masks.

-[x] Argumenting croped normal images.

-[x] Build a sequential model and train it.Then save the model.

-[x] Write a Python code that takes the DICOM image and gives its cropped images. Then, it gives these cropped images to the sequence model to predict whether these image are normal or abnormal