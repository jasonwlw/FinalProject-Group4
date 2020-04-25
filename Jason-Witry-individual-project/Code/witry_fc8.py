import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
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
#%% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 20
BATCH_SIZE = 64
DROPOUT = 0.2
EPS = 1e-8
WEIGHT_DECAY = 0
SAVE_EVERY_CHECKPOINT = 10
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



### From https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if np.random.randint(0,2) == 0:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


### transform.Normalize(torch.mean(

class HammingFilter(object):
    def __init__(self, r = 30):
        self.r = r

    def __call__(self, tensor):
        tensor = tensor.numpy()
        f = cv2.dft(np.float32(tensor[0,:,:]), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shift = np.fft.fftshift(f)
        f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
        ham = np.hamming(224)[:, None]
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** self.r
        f_filtered = ham2d*f_complex
        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
        filtered_img = np.abs(inv_img)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        im = np.zeros((224,224,3))
        im[:,:,0] = filtered_img
        im[:,:,1] = filtered_img
        im[:,:,2] = filtered_img
        #filtered_img_full = np.asarray([filtered_img,filtered_img,filtered_img]).reshape((224,224,3))
        #print(filtered_img_full.shape)
        im = Image.fromarray(im.astype(np.uint8))
        im = transforms.functional.vflip(im)
        return transforms.functional.to_tensor(im)



### Set Augmentations:

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(30,shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224,scale=(0.8,1)),
        transforms.ToTensor(),
        #HammingFilter(),
        AddGaussianNoise(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
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

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1
    #for balanced batches
    weight_per_class = [0.] * nclasses
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i]) 
    #weight_per_class = [0.25,0.25,0.5]
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# %% -------------------------------------- Train ----------------------------------------------------------

def train_model(model, criterion, optimizer, scheduler=None, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()



                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                ### Save checkpoints here
                if (epoch + 1) % SAVE_EVERY_CHECKPOINT == 0:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    is_best = False
                    save_ckp(checkpoint, is_best, CHECKPOINT_DIR, model_dir)
            if phase == 'train':
                epoch_loss_train = running_loss / dataset_sizes[phase]
                epoch_acc_train = running_corrects.double() / dataset_sizes[phase]

            elif phase == 'val':
                epoch_loss_val = running_loss / dataset_sizes[phase]
                epoch_acc_val = running_corrects.double() / dataset_sizes[phase]



            # deep copy the model
            if phase == 'val' and epoch_acc_val > best_acc:
                best_acc = epoch_acc_val
                best_model_wts = copy.deepcopy(model.state_dict())


        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


pneumonia_dataset = datasets.ImageFolder(root='../chest_xray_256x256/train',transform=data_transforms['train'])
pneumonia_val_dataset = datasets.ImageFolder(root='../chest_xray_256x256/val',transform=data_transforms['val'])

weights = make_weights_for_balanced_classes(pneumonia_dataset.imgs, len(pneumonia_dataset.classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights),replacement=True)

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(pneumonia_dataset, batch_size = BATCH_SIZE, sampler = sampler, num_workers=0)
dataloaders['val'] = torch.utils.data.DataLoader(pneumonia_val_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)

dataset_sizes = {}
dataset_sizes['train'] = len(pneumonia_dataset)
dataset_sizes['val'] = len(pneumonia_val_dataset)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 4096)
drop = nn.Dropout(DROPOUT)
act = nn.LeakyReLU()
model_ft = nn.Sequential(model_ft,
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(4096),
        nn.Linear(4096,4096),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(4096),
        nn.Linear(4096,2048),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(2048),
        nn.Linear(2048,2048),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(2048),
        nn.Linear(2048,1024),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(1024),
        nn.Linear(1024,3))
print(model_ft)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.05)

model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler = None,num_epochs=N_EPOCHS)

