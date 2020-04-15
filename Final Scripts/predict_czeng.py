import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 32, 3)
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
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)

def predict(image_paths):
    size = (28, 28)
    image_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    model = CNN().to(device)
    model.load_state_dict(torch.load("model_czeng.pt"))
    model.eval()
    # images = []
    y_pred = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image_transform(image)
        image = image.unsqueeze(0)
        y_pred.append(model(image.to(device)).cpu().detach().numpy())
    y_pred = torch.squeeze(torch.FloatTensor(y_pred), 1)
    y_pred = torch.sigmoid(y_pred)

    return y_pred.cpu()
#
# x_test = ['./train/cells_0.png', './train/cells_1.png']
#
# y_test_pred = predict(x_test)
# assert isinstance(y_test_pred, type(torch.Tensor([1])))  # Checks if your returned y_test_pred is a Torch Tensor
# assert y_test_pred.dtype == torch.float  # Checks if your tensor is of type float
# assert y_test_pred.device.type == "cpu"  # Checks if your tensor is on CPU
# assert y_test_pred.requires_grad is False  # Checks if your tensor is detached from the graph
# assert y_test_pred.shape == (len(x_test), 7)  # Checks if its shape is the right one
# # Checks whether the your predicted labels are one-hot-encoded
# #assert set(list(np.unique(y_test_pred))) in [{0}, {1}, {0, 1}]
# print("All tests passed!")