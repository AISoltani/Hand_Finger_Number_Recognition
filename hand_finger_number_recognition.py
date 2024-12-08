import os
import numpy as np
from skimage import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from torchvision.models.vgg import vgg19
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

path_train = "./train"
path_test = "./test"
image_size = 128
BATCH = 60
n_class = 6
EPOCHS = 2
device = "cuda"

class FingerNumber(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transformer = transform
        self.data = None
        self.label = None
        self.names = None
        self.__make_label()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.label[item]
        name = self.names[item]
        image = io.imread(x)
        image = np.expand_dims(image, -1).repeat(3, axis=-1)

        if self.transformer != None:
            image = self.transformer(image)
        return name, image, torch.tensor(y, dtype=torch.int)

    def __make_label(self):
        x = []
        y = []
        names = []
        for p in os.listdir(self.path):
            names.append(p)
            x.append(os.path.join(self.path, p))
            y.append(int(p.split("_")[-1][:-4]))
        self.data = x
        self.label = y
        self.names = names

def show_tensor_images(image_tensor, targets, num_images=25, ncol=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    nrow = len(image_tensor) // ncol
    fig = plt.figure(figsize=(10, 7))
    for idx, (img_tensor, lbl) in enumerate(zip(image_tensor, targets)):
        img_tensor = (img_tensor + 1) / 2
        image_unflat = img_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=nrow, value_range=(0, 5))

        fig.add_subplot(nrow, ncol, idx + 1)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.axis("off")
        plt.title(str(lbl))
    if show:
        plt.show()

def evaluation(model, criterion, val_dataloader):
    print(f"evaluation steps number {len(val_dataloader)}")
    classifier_correct = 0
    num_validation = 0
    classifier_val_loss = 0
    for img, lbl in val_dataloader:
        img = img.to(device)
        lbl = lbl.to(device)
        cur_batch_size = len(img)
        num_validation += cur_batch_size
        pred = model(img)
        classifier_val_loss += criterion(pred, lbl) * cur_batch_size
        classifier_correct += (pred.argmax(1) == lbl).float().sum()
    loss = classifier_val_loss.item() / num_validation
    acc = classifier_correct.item() / num_validation
    print(f"Test loss: {round(loss, 3)},"
    f"accuracy: {round(acc, 3)}")
    return loss, acc
     

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor()])
train_set = FingerNumber(path_train, trans)
train_len = int(train_set.__len__() * .75)
test_len = int(train_set.__len__() * .25)

train_ds, val_ds = random_split(train_set, [train_len, test_len], generator=torch.Generator().manual_seed(45))


train_data_loader = DataLoader(train_ds, BATCH, drop_last=True)
val_data_loader = DataLoader(val_ds, BATCH, drop_last=True)


_, img, lbls = next(iter(train_data_loader))
lbls = lbls.cpu().detach().numpy()
show_tensor_images(img, lbls)
del img, lbls

model = vgg19(pretrained=True).to(device)
for p in model.features.parameters():
    p.requires_grad = False

in_features = model.classifier[-4].in_features
last_layer = nn.Linear(in_features=in_features, out_features=n_class).to(device)
model.classifier[-1] = last_layer


status_step = 5

criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())
schedual = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
for e in range(EPOCHS):
    classifier_train_loss = 0
    classifier_correct = 0
    num_samples = 0
    for idx, (_, x, y) in enumerate(tqdm(train_data_loader)):
        x = x.to(device)
        y = y.to(torch.long)
        y = y.to(device)
        # y = nn.functional.one_hot(y, n_class)
        num_samples += len(x)
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        schedual.step()
        classifier_train_loss += loss.item()
        classifier_correct += (pred.argmax(1) == y).float().sum()

        if idx % status_step == 0:
             print(f"Epoch {e + 1}/{EPOCHS} step/samples {idx}/{len(train_data_loader)} Training"
                      f" loss: {round(classifier_train_loss / num_samples, 3)}"
                      f" Accuracy: {round(classifier_correct.item() / num_samples, 3)}"
                      f", lr={round(opt.param_groups[-1]['lr'], 6)}")
             classifier_train_loss = 0
             classifier_correct = 0
             num_samples = 0

    torch.save({"model": model.state_dict()}, "./vgg19.pt")
    evaluation(model, criterion, val_data_loader)

    


model = vgg19(pretrained=True).to(device)
for p in model.features.parameters():
    p.requires_grad = False

in_features = model.classifier[-4].in_features
last_layer = nn.Linear(in_features=in_features, out_features=n_class).to(device)
model.classifier[-1] = last_layer



test_set = FingerNumber(path_test, trans)
model.load_state_dict(torch.load("./vgg19.pt")["model"])
test_data_loader = DataLoader(test_set, 20, drop_last=False)

files_names = []
preds_names = []
for file_names, img, _ in test_data_loader:
    img = img.to(device)
    pred = model(img)
    
    labels = pred.argmax(1).cpu().detach().numpy()
    show_tensor_images(img, labels)
