# TOOTH INFECTION
## Description:
We wanted to predict tooth infection using opg images,

for this we first draw a box around each tooth.

Then we check these teeth individually to see if they have an infection or not.
### OPG image example.
  ![OPG image](./images/opg.png?row=true)

### Detect each tooth.
![classes](./images/yolo.png?row=true)

### Croped tooth.
![croped](./images/crop.png?row=true)


# TODO:
- [x] Read **DICOM** images and save them in png format(to train `Yolo`).
- [x] Labeling some of the images for `Yolo`.
- [x] Train **Yolo5** and save the model.
- [x] Separating normal images from abnormals.
- [x] Crop normal images with Pre-trained Yolo and save them in a new folder.
- [x] Crop abnormal images using their masks.
- [x] **Augmenting** croped normal images.
- [x] Build a **sequential** model and train it.Then save the model.
- [x] Write a Python code that takes the **DICOM** image and gives its cropped images. Then, it gives these cropped images to the sequential model to predict whether these image are normal or abnormal.
- [x] Testing the code on new data.
