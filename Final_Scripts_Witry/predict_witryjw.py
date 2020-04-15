import os

os.system("sudo pip install itertools")
os.system("sudo pip install opencv-python")
os.system("sudo pip install numpy")
os.system("sudo pip install torch")

import numpy as np
import torch
import torch.nn as nn
import cv2
from itertools import chain
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        #self.act = torch.nn.Sigmoid()
        self.drop = nn.Dropout2d(0.2)
        #self.linear1 = nn.Linear(32*5*5, 400)  # input will be flattened to (n_examples, 32 * 5 * 5)
        #self.linear1_bn = nn.BatchNorm1d(400)
        #self.drop = nn.Dropout(DROPOUT)
        #self.linear2 = nn.Linear(400, 21)
        self.act = torch.relu

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





def resize_im(im):
    desired_size = (1200,1600)

    im = cv2.resize(im, desired_size)
    return im

def image_cropping(im):
    interval = 200
    ims = []
    targs_full = []
    for rows in range(0,im.shape[0],interval):
        for cols in range(0,im.shape[1],interval):
            crop_im = im[rows:rows+interval,cols:cols+interval,:]
            ims.append(crop_im)
    return ims


# This predict is a dummy function, yours can be however you like as long as it returns the predictions in the right format
def predict(x):
    # On the exam, x will be a list of all the paths to the images of our held-out set
    images = []
    model = CNN().to(device)
    model.load_state_dict(torch.load("model_witryjw.pt"))
    model.eval()
    y_pred_full = []
    loss = 0
    criterion = nn.BCELoss()
    for i in range(0,len(x)):
        im = cv2.imread(x[i])/255.
        im = resize_im(im)
        im = image_cropping(im)
        #images = list(chain(*images))
        im = torch.FloatTensor(np.array(im)).view(len(im), 3, 200, 200).to(device)
        y_pred = model(im).squeeze()
        y_pred = torch.sigmoid(y_pred)
        #print(torch.max(y_pred,1))
        #print(torch.max(y_pred,0))
        loss += criterion(torch.max(y_pred,0)[0],torch.Tensor(y[i]).to(device)).item()
        y_pred = y_pred.cpu().detach().numpy()
        y_pred_full.append(list(np.max(y_pred, axis=0)))
    print("Loss: ",loss/len(x))
    return y_pred_full

