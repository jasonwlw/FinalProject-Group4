import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torchvision
import os
import numpy as np
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import accuracy_score
import imutils
from PIL import Image, ImageEnhance
from itertools import repeat
from itertools import chain
import json
import copy
import time
import matplotlib.pyplot as plt
#%% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 100
BATCH_SIZE = 4
DROPOUT = 0.2
EPS = 1e-8
WEIGHT_DECAY = 0
SAVE_EVERY_CHECKPOINT = 25
CHECKPOINT_DIR = './'
SAVE_EVERY_MODEL = 100
model_dir = './BEST/'

print("Learning Rate:", LR)
print("N_Epochs:", N_EPOCHS)
print("Batch_Size:", BATCH_SIZE)
print("DROPOUT:", DROPOUT)
print("EPS:", EPS)
print("WEIGHT_DECAY", WEIGHT_DECAY)


# %%--------------------------------------- Augmenting -------------------------------------------------------------




# from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return {'image': image}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return {'image': img}


### From https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if np.random.randint(0,2) == 0:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            print("HERE")
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


### transform.Normalize(torch.mean(

### Set Augmentations:

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}

# %% ----------------------------------- Helper Functions --------------------------------------------------------------


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    print("Saving checkpoint")
    if not is_best:
        f_path = checkpoint_dir + 'checkpoint.pt'
        torch.save(state, f_path)
    else:
        f_path = best_model_dir + 'checkpoint_best_model4.pt'
        torch.save(state, f_path)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# %% -------------------------------------- Test ----------------------------------------------------------



def test_model(model, criterion):

    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if epoch == 1:
                    print(labels.data)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                #optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if epoch == 1:
                        print(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss_val = running_loss / dataset_sizes[phase]
            epoch_acc_val = running_corrects.double() / dataset_sizes[phase]



            # deep copy the model

        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

        print()


    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #return model


def visualize_model(model, num_images=2):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print(preds)
            print(labels.data)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

pneumonia_dataset = datasets.ImageFolder(root='../chest_xray_256x256/train',transform=data_transforms['train'])
pneumonia_val_dataset = datasets.ImageFolder(root='../chest_xray_256x256/val',transform=data_transforms['val'])

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(pneumonia_dataset,batch_size = BATCH_SIZE, shuffle = True, num_workers=0)
dataloaders['val'] = torch.utils.data.DataLoader(pneumonia_val_dataset,batch_size = BATCH_SIZE, shuffle = True, num_workers=0)

dataset_sizes = {}
dataset_sizes['train'] = len(pneumonia_dataset)
dataset_sizes['val'] = len(pneumonia_val_dataset)
class_names = pneumonia_dataset.classes

inputs,classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 3)
model_ft.load_state_dict(torch.load("../checkpoint.pt")['state_dict'])
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
visualize_model(model_ft)


