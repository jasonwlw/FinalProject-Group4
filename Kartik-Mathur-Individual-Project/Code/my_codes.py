import torchvision
import torch
import os
import torch
import torchvision
import matplotlib.pyplot as plt
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
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,cohen_kappa_score,classification_report, confusion_matrix

dirpath = os.getcwd()  #
print("current directory is : " + dirpath)
path2add = 'FinalProject'
filepath = os.path.join(dirpath,path2add)
print("FIle path directory is : " + filepath)
def load_dataset():
    train_path = os.path.join(filepath,'chest_xray_256x256/train')
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    valid_path = os.path.join(filepath,'chest_xray_256x256/val')
    valid_dataset = torchvision.datasets.ImageFolder(
        root=valid_path,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    test_path = os.path.join(filepath,'chest_xray_256x256/test')
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_path,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
def get_confusion_results(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax,fmt='d')
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    plt.show()
    print(cm)
    Accuracy_Score = accuracy_score(true_labels, pred_labels)
    Precision_Score = precision_score(true_labels, pred_labels, average="weighted")
    Recall_Score = recall_score(true_labels, pred_labels, average="weighted")
    F1_Score = f1_score(true_labels, pred_labels)
    Cohens_kappa = cohen_kappa_score(true_labels, pred_labels)
    print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean() * 100, Accuracy_Score.std() * 100))
    print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean() * 100, Precision_Score.std() * 100))
    print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean() * 100, Recall_Score.std() * 100))
    print(' F1 Score: %0.2f +/- (%0.1f) %%' % (F1_Score))
    print(' COhen Kappa Score: %0.2f +/- (%0.1f) %%' % (Cohens_kappa))
#----------------------------------------------------------------------
print(" LOADING DEVICE")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#----------------------------------------------------------------------
print(" PARAMETERS")
num_epochs = 30
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
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32), #equal to out_channels
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2)) #reduces kernel size by /n if maxpool, n if avg pool
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), #activation Function
            # nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(in_features=32*64*64, out_features = 3) #

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        # print(" OUT ",out, out.shape)
        return out

cnn = CNN().to(device)
model2 = torchvision.models.resnet18(pretrained=True)
num_ftrs = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs, 4096)
model2 = nn.Sequential(model2,
                             nn.LeakyReLU(),
                             nn.Linear(4096, 2048),
                             nn.LeakyReLU(),
                             nn.Dropout(0.5),
                             nn.BatchNorm1d(2048),
                             nn.Linear(1024, 512),
                             nn.LeakyReLU(),
                             nn.Dropout(0.5),
                             nn.BatchNorm1d(512),
                             nn.Linear(512, 3)
                       )
model = model2.to('cuda')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    total_valid_loss = 0
    loss_train = 0
    true_labels = []
    pred_labels = []

    cnn.train().cuda()
    for i, (images, labels) in enumerate(train_loader):
        images = images.float()
        # torch(images)
        labels = labels.long()
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = cnn(images)
        _, train_preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        item = loss.item()
        loss_train += item # loss for each batch

        if (i + 1) % 25 == 0:
            print('Epoch [%d/%d], Iter [%d/%d]  Batch Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader) // batch_size, loss_train))
            torch.cuda.empty_cache()
        true_labels.append(labels.detach().cpu())
        pred_labels.append(train_preds.detach().cpu())

    torch.cuda.empty_cache()
    with torch.no_grad():
        for images1, labels1 in valid_loader:
                cnn.eval().cuda()
                images1 = images1.float().cuda()
                labels1 = labels1.long().cuda()
                output = cnn(images1)
                _, val_preds = torch.max(output, 1)

                loss1 = criterion(output, labels1)
                loss_valid = loss1.item() #gets loss for each batch
                total_valid_loss += loss_valid # adds loss for all batches

                # print(" Test Loss ", loss_test)

    print("Epoch",epoch+1," Total Train loss:", loss_train,"Total Validation Loss",total_valid_loss,'\n')
# -----------------------------------------------------------------------------------



