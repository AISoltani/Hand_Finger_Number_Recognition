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

# Code Explanation
1. FingerNumber Dataset Class
This class is used to load and preprocess the images. Each image is associated with a label, which is extracted from the file name. The image is loaded using skimage.io.imread, then expanded to have 3 color channels. The images are then transformed (resized and converted to tensor format) before being returned for model training.

```bash
class FingerNumber(Dataset):
    def __init__(self, path, transform=None):
        ...
```
2. show_tensor_images Function
This function visualizes the images and their corresponding labels in a grid format. It is useful for inspecting a batch of images during training or evaluation.

```bash
def show_tensor_images(image_tensor, targets, num_images=25, ncol=5, show=True):
```
3. evaluation Function
The evaluation function computes the loss and accuracy of the model on a given validation or test dataset. It uses the CrossEntropy loss and compares the predicted labels with the true labels.

```bash
def evaluation(model, criterion, val_dataloader):
    ...
```
4. Model Setup
A pretrained VGG19 model is used. The convolutional layers are frozen, and the final fully connected layer is replaced with a custom classifier that outputs six classes (for the six finger numbers).

```bash
model = vgg19(pretrained=True).to(device)
for p in model.features.parameters():
    p.requires_grad = False

in_features = model.classifier[-4].in_features
last_layer = nn.Linear(in_features=in_features, out_features=n_class).to(device)
model.classifier[-1] = last_layer
```
5. Training
The model is trained using the Adam optimizer with a learning rate scheduler (CosineAnnealingLR). The training loop updates the model weights using backpropagation and evaluates the model periodically on the validation set.

```bash
for e in range(EPOCHS):
    for idx, (_, x, y) in enumerate(tqdm(train_data_loader)):
        ...
```
6. Model Saving and Loading
After training, the model's state_dict is saved to ./vgg19.pt. The trained model is then loaded for testing.

```bash
torch.save({"model": model.state_dict()}, "./vgg19.pt")
```
7. Testing
The model is evaluated on the test set, and predictions are visualized for inspection. The results show the predicted class for each image in the test set.

```bash
test_set = FingerNumber(path_test, trans)
model.load_state_dict(torch.load("./vgg19.pt")["model"])
```

# Running the Code
Prepare your dataset in the ./train and ./test directories.
Set the parameters (batch size, number of epochs, image size, etc.) in the script as required.
Run the script to train the model, evaluate it, and test it.
```bash
python train_finger_recognition.py
```

# Training Configuration
Image Size: 128x128
Batch Size: 60
Number of Classes: 6 (corresponding to finger numbers 0-5)
Epochs: 2
Device: CUDA (GPU acceleration)
