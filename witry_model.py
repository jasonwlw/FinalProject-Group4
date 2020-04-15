import numpy as np
import torch
import torch.nn as nn
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
#%% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 100
BATCH_SIZE = 60
DROPOUT = 0.2
EPS = 1e-8
WEIGHT_DECAY = 0
SAVE_EVERY_CHECKPOINT = 20
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
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



### transform.Normalize(torch.mean(

# %% ----------------------------------- Helper Functions --------------------------------------------------------------


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    print("Saving checkpoint")
    if not is_best:
        f_path = checkpoint_dir + 'checkpoint.pt'
        torch.save(state, f_path)
    else:
        f_path = best_model_dir + 'checkpoint_best_model4.pt'
        torch.save(state, f_path)


def y_generator(num_images):
    ### num_images is batchsize
    return np.concatenate([['NORMAL']*num_images,['VIRUS']*num_images,['BACTERIA']*num_images])


#%% ---------------------------------------------------Data Prep -------------------------------------------------------
### Data loader class goes here - Jane


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    '''
    cfCNN: d'Acremont 2019
    '''
    def __init__(self):
        super(CNN, self).__init__()
        #input 200x200
        self.conv11 = nn.Conv2d(3, 48, (9, 9))  # output (ne, 48, 192 , 192)

        self.convnorm11 = nn.BatchNorm2d(48)
        #self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 13, 13)
        self.conv12 = nn.Conv2d(48, 48, (9, 9))  # output (n_examples, 48, 184, 184)
        self.convnorm12 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 48, 92, 92)
        self.conv21 = nn.Conv2d(48, 96, (5, 5)) # output (ne, 96, 88, 88)
        self.convnorm21 = nn.BatchNorm2d(96)
        self.conv22 = nn.Conv2d(96, 96, (5, 5)) # output (ne, 32, 84, 84)
        self.convnorm22 = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d((2, 2)) # output (ne, 32, 42, 42)
        self.conv31 = nn.Conv2d(96, 192, (5, 5)) # (ne, 192, 38, 38)
        self.convnorm31 = nn.BatchNorm2d(192)
        self.conv32 = nn.Conv2d(192, 192, (3, 3)) # output (ne, 192, 36, 36)
        self.convnorm32 = nn.BatchNorm2d(192)
        self.pool3 = nn.MaxPool2d((2, 2)) # output (ne, 64, 18, 18)
        self.conv4 = nn.Conv2d(192, 192, (3, 3)) # output (ne, 192, 16, 16)
        self.convnorm4 = nn.BatchNorm2d(192)
        self.pool4 = nn.MaxPool2d((2, 2)) # output (ne,  192, 8, 8)
        self.conv5 = nn.Conv2d(192, 7, (3, 3)) #output (ne, 7, 6, 6)
        self.convnorm5 = nn.BatchNorm2d(7)
        self.gap = nn.AvgPool2d((6,6))
        self.drop = nn.Dropout2d(DROPOUT)
        self.act = nn.LeakyReLU()
        #self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.convnorm11(self.act(self.drop(self.conv11(x))))
        x = self.pool1(self.convnorm12(self.act(self.drop(self.conv12(x)))))
        x = self.convnorm21(self.act(self.drop(self.conv21(x))))
        x = self.pool2(self.convnorm22(self.act(self.drop(self.conv22(x)))))
        x = self.convnorm31(self.act(self.drop(self.conv31(x))))
        x = self.pool3(self.convnorm32(self.act(self.drop(self.conv32(x)))))
        x = self.pool4(self.convnorm4(self.act(self.drop(self.conv4(x))))) ## 1400,1400
        #x = self.convnorm4(self.drop(self.conv4(x))) #700,700
        x = self.gap(self.convnorm5(self.act(self.conv5(x))))
        return x
        #x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        #x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        #x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        #return self.linear2(x)


# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=EPS,weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()


# %% -------------------------------------- Train ----------------------------------------------------------
print("Starting training loop...")
model.train()
shuf = np.arange(len(X_path_full))

global_loss_test = 100
last_loss_test = 100
for epoch in range(N_EPOCHS):

    loss_train = 0
    #model.train()
    for batch in range(len(X_train)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(X_train[inds]).squeeze()
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test).squeeze()
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train/BATCH_SIZE, acc(X_train, y_train), loss_test, acc(X_test, y_test)))

    ### move model train here to save a model that can be resumed; it must be in training mode to resume
    if (epoch + 1) % SAVE_EVERY_MODEL == 0:
        print("Saving model...")
        torch.save(model.state_dict(), "./model4_augmented_cropped_epochs_tanh"+str(epoch)+".pt")

    model.train()

    if (epoch + 1) % SAVE_EVERY_CHECKPOINT == 0: 
        is_best = False
        if loss_test < last_loss_test:
            if loss_test < global_loss_test:
                global_loss_test = loss_test
                is_best = True

        last_loss_test = loss_test
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, is_best, CHECKPOINT_DIR, model_dir)


