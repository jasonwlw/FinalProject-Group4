
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import copy
from torch.utils.data.sampler import WeightedRandomSampler

import os
from PIL import Image
from PIL import ImageEnhance
import cv2

# %%

# %%
import matplotlib.pyplot as plt
import matplotlib as mpb

# %%
print("Parameters")
batch_size = 128
LR = 5e-2
decay = 0.001
EPOCHS = 15
decay = 1e-3
gamma = 0.1 # LR step scheduler

np.random.seed(42)
transformers = {'train': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])}
modes = ['train', 'test', 'val']

# %%

datasets = {mode: ImageFolder('../' + mode, transformers[mode]) for mode in modes}
datasets_sizes = {mode: len(datasets[mode]) for mode in modes}

# %%

# in train: 1342 normal, 2531 bacteria, 1346 virus
# using sampler, but I overfit easily  when using sampler
# weights = [1. / 1342, 1. / 2531, 1. / 1346]
# sampler = WeightedRandomSampler(weights, num_samples=int(0.7 * len(datasets['train'])), replacement=True)
dataloaders = {}
for mode in modes:
    if mode == 'train':
        # dataloaders[mode] = DataLoader(datasets[mode], batch_size=batch_size, sampler=sampler)
        dataloaders[mode] = DataLoader(datasets[mode], batch_size=batch_size)
    else:
        dataloaders[mode] = DataLoader(datasets[mode], batch_size=batch_size)

#———————————————————————————————————— no sampler————————————————————————————————————————————————————————————————————

# not using sampler
dataloaders = {mode: DataLoader(datasets[mode], batch_size=batch_size, shuffle=True) for mode in modes}

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
print(datasets['train'].class_to_idx.keys())
print(datasets_sizes)


model = torchvision.models.resnet34(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 3)
model = model.cuda()
model

# %%

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=decay)
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma= gamma)

# %%

best_model = None
best_loss = 0.
best_test_loss = 0.
best_test_acc = 0.
best_pred_labels = []
test_true_labels = []
test_pred_labels = []
train_true_labels = []
train_pred_labels = []
test_acc = 0.
test_loss = 0.


# %%

def fit(model, dataloader, criterion, optimizer, mode='train'):
    epoch_loss = 0.
    epoch_acc = 0.

    batch_num = 0.
    samples_num = 0.

    true_labels = []
    pred_labels = []

    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.cuda(), labels.cuda()

        optimizer.zero_grad()
        with torch.set_grad_enabled(mode == 'train'):
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        true_labels.append(labels.detach().cpu())
        pred_labels.append(preds.detach().cpu())

        if mode == 'train':
            loss.backward()
            optimizer.step()
        #             scheduler.step()

        print(f'\r{mode} batch [{batch_idx}/{len(dataloader)}]: loss {loss.item()}', end='', flush=True)
        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels.data)
        batch_num += 1
        samples_num += len(labels)
    print()
    return epoch_loss / batch_num, epoch_acc.double() / samples_num, torch.cat(true_labels).numpy(), torch.cat(
        pred_labels).numpy()


# %%

train_losses = []
val_losses = []
test_losses = []
train_accs = []
val_accs = []
test_accs = []

for epoch in range(EPOCHS):
    print('=' * 15, f'Epoch: {epoch}','=' * 15)

    train_loss, train_acc, train_true_labels, train_pred_labels = fit(model, dataloaders['train'], criterion, optimizer)
    val_loss, val_acc, _, _ = fit(model, dataloaders['val'], criterion, optimizer, mode='val')
    test_loss, test_acc, test_true_labels, test_pred_labels = fit(model, dataloaders['test'], criterion, optimizer,
                                                                  mode='test')

    print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')
    print(f'Val loss: {val_loss}, Val accuracy: {val_acc}')
    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
    print()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    if best_model is None or val_loss < best_loss:
        best_model = copy.deepcopy(model)
        best_loss = val_loss
        best_test_loss = test_loss
        best_test_acc = test_acc
        best_pred_labels = test_pred_labels

torch.save({'epoch': epoch+1, 'model': model.state_dict()}, f'resnet18-chest-x-ray-best.pt')

# %%

plt.figure(figsize=(18, 8))
plt.title('Model Loss')
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.show()

# %%

plt.figure(figsize=(18, 8))
plt.title('Model Accuracy')
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.legend()
plt.show()

# %%

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(test_true_labels, best_pred_labels)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
