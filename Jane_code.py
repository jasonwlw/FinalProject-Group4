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
        
        print(f'\r{mode} batch [{batch_idx}/{len(dataloader)}]: loss {loss.item()}', end='', flush=True)
        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels.data)
        batch_num += 1
        samples_num += len(labels)
    print()
    return epoch_loss / batch_num, epoch_acc.double() / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy()

if __name__ == "__main__":
    transformers = {'train' : transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])}
    modes = ['train', 'test', 'val']

    datasets = {mode: ImageFolder('../'+mode, transformers[mode]) for mode in modes}
    dataloaders = {mode: DataLoader(datasets[mode], batch_size=32, shuffle=True) for mode in modes}
    datasets_sizes = {mode: len(datasets[mode]) for mode in modes}

    model = torchvision.models.resnet18(pretrained=True)      
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256,3))
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.001)

    best_model = None
    best_loss = 0.
    best_test_loss = 0.
    best_test_acc = 0.
    best_pred_labels = []
    true_labels = []
    pred_labels = []
    test_acc = 0.
    test_loss = 0.
    batch_size = 32

    train_losses = []
    val_losses = []
    test_losses = []
    train_accs = []
    val_accs = []
    test_accs = []

    for epoch in range(10):
        print('='*15, f'Epoch: {epoch}')
        
        train_loss, train_acc, _, _ = fit(model, dataloaders['train'], criterion, optimizer)
        val_loss, val_acc, _, _ = fit(model, dataloaders['val'], criterion, optimizer, mode='val')
        test_loss, test_acc, true_labels, pred_labels = fit(model, dataloaders['test'], criterion, optimizer, mode='test')
        
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
            best_pred_labels = pred_labels

    torch.save({'epoch': epoch, 'model': model.state_dict()}, f'resnet18-chest-x-ray-best.pt')

    
