import torchvision
import torch
import os
import torch
import torchvision
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.autograd import Variable
dirpath = os.getcwd()  #
print("current directory is : " + dirpath)
path2add = 'FinalProject'
filepath = os.path.join(dirpath,path2add)
print("FIle path directory is : " + filepath)
"/home/ubuntu/Deep-Learning/FinalProject/chest_xray_200x200/"
def load_dataset():
    train_path = os.path.join(filepath,'chest_xray_200x200/train')
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=torchvision.transforms.ToTensor()
    )

    valid_path = os.path.join(filepath,'chest_xray_200x200/val')
    valid_dataset = torchvision.datasets.ImageFolder(
        root=valid_path,
        transform=torchvision.transforms.ToTensor()
    )
    test_path = os.path.join(filepath,'chest_xray_200x200/test')
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_path,
        transform=torchvision.transforms.ToTensor()
    )
    return train_dataset,valid_dataset,test_dataset
def get_loaders(train_dataset,valid_dataset,test_dataset,batch_size):
    workers = 2
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True
        )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True
        )
    return train_loader,valid_loader,test_loader
#----------------------------------------------------------------------
print(" LOADING DEVICE")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#----------------------------------------------------------------------
print(" PARAMETERS")
num_epochs = 5
batch_size = 64
learning_rate = 0.001
#----------------------------------------------------------------------
train_dataset,valid_dataset,test_dataset = load_dataset()

#One hot encoding
for data1 in [train_dataset,valid_dataset,test_dataset]:
    le = OneHotEncoder(sparse=False)
    print(data1,'\n',data1.targets)
    integer_encoded = np.array(data1.targets).reshape(len(data1.targets), 1)
    data1.targets = le.fit_transform(integer_encoded)
    print(data1.targets)


train_loader,valid_loader,test_loader = get_loaders(train_dataset,valid_dataset,test_dataset,batch_size=batch_size)
#----------------------------------------------------------------------

print(" CNN CLASS")
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(32), #equal to out_channels
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2)) #reduces kernel size by /n if maxpool, n if avg pool
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(), #activation Function
            # nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(in_features=32*48*48, out_features = 3) #

    def forward(self, x):
        out = self.layer1(x)
        # print("LAYER 1 OUT ",out.shape)
        out = self.layer2(out)
        # print("LAYER 2 OUT ",out.shape)
        # print("OUT SIZE ",out.size(0))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # print(" OUT ",out, out.shape)
        # return (torch.sigmoid(out)) # if i dont put torch.sigmoid(out), the tensors returned are are same as the labels #bbut they still give a loss
        # out = self.fc1(out)
        return out

cnn = CNN().to(device)
# cnn.cuda()
print("  Loss and Optimizer")
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

print(" TRAINING LOOP")
# Train the Model
for epoch in range(num_epochs):
    total_valid_loss = 0
    loss_train = 0
    cnn.train().cuda()
    for i, (images, labels) in enumerate(train_loader):
        images = images.float()
        # torch(images)
        labels = labels.long()
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        "Optimizer"
        optimizer.zero_grad()
        # print("OUTPUTS")
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        # print("LOSS BACKWARD")
        loss.backward()
        # print("OPTIMIZER STEP")
        optimizer.step()
        item = loss.item()
        loss_train += item # loss for each batch

        if (i + 1) % 25 == 0:
            print('Epoch [%d/%d], Iter [%d/%d]  Batch Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader) // batch_size, loss_train))
            torch.cuda.empty_cache()

            # Test the Model
    torch.cuda.empty_cache()
    with torch.no_grad():
        for images1, labels1 in valid_loader:
                # print("Validation Loop")
                cnn.eval().cuda()
                images1 = images1.float().cuda()
                labels1 = labels1.long().cuda()
                output = cnn(images1)
                # print(type(output))
                loss1 = criterion(output, labels1)
                loss_valid = loss1.item() #gets loss for each batch
                total_valid_loss += loss_valid # adds loss for all batches

                # print(" Test Loss ", loss_test)
    print("Epoch",epoch+1," Total Train loss:", loss_train,"Total Validation Loss",total_valid_loss,'\n')
# -----------------------------------------------------------------------------------



