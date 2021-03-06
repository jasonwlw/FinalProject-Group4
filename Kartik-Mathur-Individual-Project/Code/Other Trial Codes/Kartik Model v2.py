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
from sklearn.metrics import confusion_matrix
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import time
import os


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
            # print(outputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        true_labels.append(labels.detach().cpu())
        pred_labels.append(preds.detach().cpu())

        if mode == 'train':
            loss.backward()
            optimizer.step()

        print(f'\r{mode} batch [{batch_idx}/{len(dataloader)}]: loss {loss.item()}', end='', flush=True)
        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels.data)
        batch_num += 1
        samples_num += len(labels)
    print()
    return epoch_loss / batch_num, epoch_acc.double() / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy()

def get_confusion_results(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax);  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    plt.show()
    Accuracy_Score = accuracy_score(true_labels, pred_labels)
    Precision_Score = precision_score(true_labels, pred_labels, average="macro")
    Recall_Score = recall_score(true_labels, pred_labels, average="macro")
    print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean() * 100, Accuracy_Score.std() * 100))
    print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean() * 100, Precision_Score.std() * 100))
    print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean() * 100, Recall_Score.std() * 100))


if __name__ == "__main__":
    print("Parameters")
    batch_size = 128
    LR = 5e-3
    decay = 0.001
    EPOCHS = 30
    dropout = 0.3
    resize = 256


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("LOADING DEVICE:", device)

    print("Transforms")
    transformers = {'train' : transforms.Compose([
    transforms.Resize(resize),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test' : transforms.Compose([
        transforms.Resize(resize),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize(resize),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])}

    modes = ['train', 'test', 'val']
    print("Datasets")
    datasets = {mode: ImageFolder('FinalProject/chest_xray_256x256/'+mode, transformers[mode]) for mode in modes}
    dataloaders = {mode: DataLoader(datasets[mode], batch_size=batch_size, shuffle=True) for mode in modes}
    datasets_sizes = {mode: len(datasets[mode]) for mode in modes}


    print("Models")
    model = torchvision.models.resnet34(pretrained=True)
    # model1 = torchvision.models.VGG16(pretrained=True)
    # model.classifier[16] = nn.Linear(1000,3)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 16),
        nn.LeakyReLU(),
        nn.Dropout2d(dropout),
        # nn.Conv2d(in_channels=256,out_channels=16,kernel_size=2,padding=0),
        # nn.LeakyReLU(),
        # nn.MaxPool2d(kernel_size=2),
        # nn.Linear(256, 16),
        nn.Linear(16,3)
        )
    model = model.to('cuda')
    print("Critereon & Optimizer")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=decay)
    # final_model = nn.Sequential(model,model1)
    best_model = None
    best_loss = 0.
    best_test_loss = 0.
    best_test_acc = 0.
    best_pred_labels = []
    true_labels = []
    pred_labels = []
    test_acc = 0.
    test_loss = 0.
    batch_size = batch_size

    train_losses = []
    val_losses = []
    test_losses = []
    train_accs = []
    val_accs = []
    test_accs = []

    for epoch in range(EPOCHS):
        print('='*15, f'Epoch: {epoch+1}','='*15)

        train_loss, train_acc, _, _ = fit(model, dataloaders['train'], criterion, optimizer)
        val_loss, val_acc, _, _ = fit(model, dataloaders['val'], criterion, optimizer, mode='val')
        test_loss, test_acc, true_labels, pred_labels = fit(model, dataloaders['test'], criterion, optimizer, mode='test')

        print(f'    Train loss: {train_loss},  Train accuracy: {train_acc}')
        print(f'    Val loss  : {val_loss},    Val accuracy  : {val_acc}')
        print(f'    Test loss : {test_loss},   Test accuracy : {test_acc}')
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
            best_pred_labels = pred_labels
    get_confusion_results(true_labels, pred_labels)
    torch.save({'epoch': epoch, 'model': model.state_dict()}, f'resnet18-chest-x-ray-best.pt')

