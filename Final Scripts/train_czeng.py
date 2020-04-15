import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#----------------------------------------------------------------------------


class ImageData(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.images_f = [f for f in os.listdir(self.root_dir) if os.path.splitext(f)[1] == '.png']
        self.mode = mode
        self.labels_dic = {"red blood cell":0, "difficult":1, "gametocyte":2, "trophozoite":3,
                          "ring":4, "schizont":5, "leukocyte":6}

        if self.mode == 'train':
            self.labels_f = [f for f in os.listdir(self.root_dir) if os.path.splitext(f)[1] == '.txt']
        
    def __len__(self):
        return len(self.images_f)
    
    def __getitem__(self, index):
        image_index = self.images_f[index]
        image_path = os.path.join(self.root_dir, image_index)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            label_index = self.labels_f[index]
            label_path = os.path.join(self.root_dir, label_index)
            label = self.get_label_one_hot(label_path)
            return image, label
        
        elif self.mode == 'test':
            return image

    def get_label_one_hot(self, label_path):
        labels = open(label_path, 'r').read().split('\n')
        label_one_hot = np.zeros(7)
        for i in labels:
            label_one_hot[self.labels_dic[i]] = 1
        return label_one_hot

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2))
        self.linear1 = nn.Linear(32*5*5, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(400, 7)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.act(self.linear1(x.view(len(x), -1)))
        x = self.drop(self.linear1_bn(x))
        return self.linear2(x)



def train():
    size = (28, 28)
    image_transform = transforms.Compose([
        transforms.Scale(size),
        transforms.ToTensor()
    ])
    data = ImageData('./train', transform=image_transform)
    dataloader = DataLoader(data, batch_size=256, shuffle=False, drop_last=True)

    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
    criterion = nn.BCEWithLogitsLoss()

    # training
    for epoch in range(50):
        model.train()
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0: 
            print(epoch, " epoch train ends.")
    
    torch.save(model.state_dict(), "model_czeng.pt")

train()
# model.load_state_dict(torch.load("model_czeng.pt"))
# model.eval()
# print(model)
