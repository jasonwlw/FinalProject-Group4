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

np.random.seed(42)
# %%-----------------------------------------------------Helper Functions ---------------------------------------

# def read_data(datapath='../train/',test_size = 0.2,num_data = None,num_filetypes=1):
#     im_list = []
#     num_files = 0
#     dir = sorted(os.listdir(datapath))
#     for fil in dir:
#         if fil.endswith('.png') or fil.endswith('.jpeg') or fil.endswith('.jpg'):
#             im_list.append(datapath+fil)
#         num_files += 1
#         ### num_data = None gives all of the images
#         if num_data is not None:
#             ### How many file types are in the folder? If there are 2, i.e. .png and .txt,
#             ### We will need to go through 20 files to get 10 images and 10 targets
#             if num_files >= num_data*num_filetypes:
#                 break
#         else:
#             pass
#     #im_list,y_list = align_data(im_list,y_list)
#     return np.asarray(im_list)
#
# def resize_im(im):
#     desired_size = (256,256)
#     return cv2.resize(im,desired_size)
#
# def align_data(im_list):
#     ### Find type of pneumonia from image path
#     pneu_type = [im.split('_')[2] for im in im_list]
#     virus = []
#     bacteria = []
#     for i in range(len(pneu_type)):
#         if pneu_type[i] == 'virus':
#             virus.append(im_list[i])
#         elif pneu_type[i] == 'bacteria':
#             bacteria.append(im_list[i])
#     return np.asarray(virus),np.asarray(bacteria)
#
# def batch_generator(normal,virus,bacteria,num_images):
#     norm = np.random.choice(normal,num_images,replace = False)
#     vir = np.random.choice(virus,num_images,replace = False)
#     bact = np.random.choice(bacteria,num_images,replace = False)
#     return np.concatenate([norm, vir, bact])
#
# # %% ------------------------------- Read Data -------------------------------------------------------------------
# ### enter the paths to the data on your system
# filepaths_all = ['../chest_xray/train/NORMAL/','../chest_xray/train/PNEUMONIA/',
#              '../chest_xray/val/NORMAL/','../chest_xray/val/PNEUMONIA/',
#              '../chest_xray/test/NORMAL/','../chest_xray/test/PNEUMONIA/']
#
# keys = ['train','val','test']
# all_folders = {}
# for key in keys:
#     filepaths = [path for path in filepaths_all if key in path]
#     all_types = {}
#     for filepath in filepaths:
#         X = read_data(datapath=filepath, num_data=None)
#         if filepath.split('/')[-2] == 'PNEUMONIA':
#             virus,bacteria = align_data(X)
#             all_types['virus'] = virus
#             all_types['bacteria'] = bacteria
#
#         else:
#             all_types['normal'] = X
#             X = None
#     all_folders[key] = all_types
# ### 5,863 images * 0.15 in validation gives ~ 830 images
# ### Nearest divisible by 3 is 831, gives 277 of each image
# ### There are 8 bacterial and 8 normal in validation
# ### Num bacteria and normal = 277 - 8 = 269
# ### num virus is 277
# ### Magic numbers will ensure this does not work if read_data does not read in more data than the numbers
# ### above
# ###
# train_vir = np.random.choice(all_folders['train']['virus'],size=277,replace=False)
# #train_vir = np.random.choice(all_folders['train']['virus'],size=2,replace=False)
# ### remove from train
# all_folders['train']['virus'] = np.setdiff1d(all_folders['train']['virus'],train_vir)
# ### put in validation
# all_folders['val']['virus'] = np.append(all_folders['val']['virus'],train_vir)
#
# train_bact = np.random.choice(all_folders['train']['bacteria'],size=269,replace=False)
# #train_bact = np.random.choice(all_folders['train']['bacteria'],size=2,replace=False)
# ### remove from train
# all_folders['train']['bacteria'] = np.setdiff1d(all_folders['train']['bacteria'],train_bact)
# ### put in validation
# all_folders['val']['bacteria'] = np.append(all_folders['val']['bacteria'],train_bact)
#
# train_normal = np.random.choice(all_folders['train']['normal'],size=269,replace=False)
# #train_normal = np.random.choice(all_folders['train']['normal'],size=2,replace=False)
# ### remove from train
# all_folders['train']['normal'] = np.setdiff1d(all_folders['train']['normal'],train_normal)
# ### put in validation
# all_folders['val']['normal'] = np.append(all_folders['val']['normal'],train_normal)
#
#
#
# ### dict for easy access to each array
#
# ### Loop through dictionary keys
# for folder in all_folders:
#     all_types = all_folders[folder]
#     for types in all_types:
#     ### Loop through file paths in the array
#         print("Folder",folder)
#         print("Class",types)
#         for filepath in all_types[types]:
#             im = cv2.imread(filepath)
#             im = resize_im(im)
#             ### numpy is binary file can save space/maybe load time
#             #np.save(filepath.replace('chest_xray','chest_xray_200x200').replace('.jpeg',''),im)
#             savepath = filepath.replace('chest_xray','chest_xray_256x256')
#             if folder == 'val':
#                 savepath = savepath.replace('train','val')
#             ### Alternatively, if your script is in the chest_xray folder:
#             #savepath = filepath.replace('train','train_200x200')
#             ### You may want to further edit the above for your system, or comment it out to overwrite the
#             ### downloaded files.
#
#             ### Quick and dirty edit to save all at once, make necessary directories.
#             ### can comment out resizing if you want full images, or change the desired size in the resize im function.
#             if types == 'normal':
#                 dirpath = '/'.join(savepath.split('/')[:-1])
#                 os.system('mkdir -p ' + str(dirpath))
#                 cv2.imwrite(savepath,im)
#             elif types == 'virus':
#                 ### Replace PNEUMONIA directory with PNEUMONIA-VIRAL
#                 savepath = savepath.replace('PNEUMONIA','PNEUMONIA-VIRAL')
#                 dirpath = '/'.join(savepath.split('/')[:-1])
#                 os.system('mkdir -p ' + str(dirpath))
#                 cv2.imwrite(savepath, im)
#             elif types == 'bacteria':
#                 savepath = savepath.replace('PNEUMONIA','PNEUMONIA-BACTERIAL')
#                 dirpath = '/'.join(savepath.split('/')[:-1])
#                 os.system('mkdir -p ' + dirpath)
#                 cv2.imwrite(savepath,im)
#             else:
#                 print("Something went terribly wrong")
#

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
weights = [1. / 1342, 1. / 2531, 1. / 1346]
sampler = WeightedRandomSampler(weights, num_samples=int(0.7 * len(datasets['train'])), replacement=True)
dataloaders = {}
for mode in modes:
    if mode == 'train':
        dataloaders[mode] = DataLoader(datasets[mode], batch_size=16, sampler=sampler)
    else:
        dataloaders[mode] = DataLoader(datasets[mode], batch_size=16)

#———————————————————————————————————— no sampler————————————————————————————————————————————————————————————————————

# not using sampler
# dataloaders = {mode: DataLoader(datasets[mode], batch_size=4, shuffle=True) for mode in modes}

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
print(datasets['train'].class_to_idx.keys())
print(datasets_sizes)


model = torchvision.models.resnet34(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 3)
model = model.cuda()
model

# %%

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=5e-2, weight_decay=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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

for epoch in range(20):
    print('=' * 15, f'Epoch: {epoch}')

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

torch.save({'epoch': epoch, 'model': model.state_dict()}, f'resnet18-chest-x-ray-best.pt')

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

# %%


