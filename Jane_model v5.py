
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


import matplotlib.pyplot as plt
import matplotlib as mpb




transformers = {'train' : transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=1.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
'test' : transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
'val' : transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])}
modes = ['train', 'test', 'val']


datasets = {mode: ImageFolder('../'+mode, transformers[mode]) for mode in modes}
datasets_sizes = {mode: len(datasets[mode]) for mode in modes}




dataloaders = {mode: DataLoader(datasets[mode], batch_size = 4, shuffle=True) for mode in modes}



model1 = torchvision.models.inception_v3(aux_logits=False)
for param in model1.parameters():
    param.requires_grad = False
model1.fc = nn.Linear(model1.fc.in_features, 3)
model1 = model1.cuda()



model2 = torchvision.models.vgg16()
for param in model2.parameters():
    param.requires_grad = False
model2.classifier = nn.Sequential(nn.Linear(model2.classifier[0].in_features, 4096),
                                  nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Linear(4096,4096),
                                  nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Linear(4096, 3))
model2 = model2.cuda()



criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr = 1e-3)
optimizer2 = torch.optim.Adam(model2.parameters(),lr = 1e-3)
lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=3)
lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=3)
weight1 = 0.4
weight2 = 0.6
epochs = 10

models = [model1, model2]
criterions = [criterion1, criterion2]
optimizers = [optimizer1, optimizer2]
weights = [weight1, weight2]
lr_schedulers = [lr_scheduler1, lr_scheduler2]

best_model1 = None
best_model2 = None
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


def fit(models, dataloader, criterions, optimizers, weights, mode='train'):
    epoch_loss = 0.
    epoch_acc = 0.
    
    batch_num = 0.
    samples_num = 0.
    
    true_labels = []
    pred_labels = []
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.cuda(), labels.cuda()
        
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()
        with torch.set_grad_enabled(mode == 'train'):
            outputs1 = models[0](data)
            outputs2 = models[1](data)
            outputs = weights[0] * outputs1 + weights[1] * outputs2
            _, preds = torch.max(outputs, 1)
            
            loss1 = criterions[0](outputs1, labels)
            loss2 = criterions[1](outputs2, labels)
        
        true_labels.append(labels.detach().cpu())
        pred_labels.append(preds.detach().cpu())
        
        if mode == 'train':
            loss1.backward()
            loss2.backward()
            optimizers[0].step()
            optimizers[1].step()
            
        print(f'\r{mode} batch [{batch_idx}/{len(dataloader)}]: loss {weights[0]*loss1.item()+weights[1]*loss2.item()}', end='', flush=True)
        epoch_loss += (weights[0]*loss1.item()+weights[1]*loss2.item())
        epoch_acc += torch.sum(preds == labels.data)
        batch_num += 1
        samples_num += len(labels)
    print()
    return epoch_loss / batch_num, epoch_acc.double() / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy()



train_losses = []
val_losses = []
test_losses = []
train_accs = []
val_accs = []
test_accs = []

for epoch in range(epochs):
    print('='*15, f'Epoch: {epoch}')
    
    train_loss, train_acc, train_true_labels, train_pred_labels = fit(models, dataloaders['train'], criterions, optimizers, weights)
    val_loss, val_acc, _, _ = fit(models, dataloaders['val'], criterions, optimizers, weights, mode='val')
    test_loss, test_acc, test_true_labels, test_pred_labels = fit(models, dataloaders['test'], criterions, optimizers,weights,mode='test')
    
    
#     lr_schedulers[0].step()
#     lr_schedulers[1].step()
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
    
    if test_loss < best_loss:
            best_model1 = copy.deepcopy(model1)
            best_model2 = copy.deepcopy(model2)
            best_loss = test_loss
            best_test_loss = test_loss
            best_test_acc = test_acc 
            best_pred_labels = test_pred_labels

torch.save({'epoch': epoch, 'model': model1.state_dict()}, f'inceptionv3-chest-x-ray-best.pt')
torch.save({'epoch': epoch, 'model': model2.state_dict()}, f'vgg16-chest-x-ray-best.pt')



plt.figure(figsize=(18, 8))
plt.title('Model Loss')
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.show()



plt.figure(figsize=(18, 8))
plt.title('Model Accuracy')
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(test_true_labels, best_pred_labels)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()






