# Hand_Finger_Number_Recognition
Predict Hand Finger Figure Number
This repository implements a finger recognition system using the VGG19 model, based on a custom dataset of images labeled with finger counts. The model is trained to classify images into one of six possible categories corresponding to finger numbers.

# Requirements
Make sure to install the following dependencies:

- Python 3.x
- NumPy
- PyTorch
- Scikit-Image
- TorchVision
- Matplotlib
- TQDM
You can install the required packages using pip
```bash
pip install numpy torch torchvision scikit-image matplotlib tqdm
```
# Dataset
The dataset is assumed to be stored in the following directories:

- ./train - Contains training images, each named in the format: image_#.jpg, where # corresponds to the label (finger number).
- ./test - Contains test images with the same naming convention.
The dataset should be organized such that each image is in a folder corresponding to its label.

Code Explanation
1. FingerNumber Dataset Class
This class is used to load and preprocess the images. Each image is associated with a label, which is extracted from the file name. The image is loaded using skimage.io.imread, then expanded to have 3 color channels. The images are then transformed (resized and converted to tensor format) before being returned for model training.
