import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
import numpy as np
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import accuracy_score
import imutils
from PIL import Image, ImageEnhance
from itertools import repeat
from itertools import chain
import json
import copy
import time
#%% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 100
BATCH_SIZE = 60
DROPOUT = 0.2
EPS = 1e-8
WEIGHT_DECAY = 0
SAVE_EVERY_CHECKPOINT = 25
CHECKPOINT_DIR = './'
SAVE_EVERY_MODEL = 100
model_dir = './BEST/'

print("Learning Rate:", LR)
print("N_Epochs:", N_EPOCHS)
print("Batch_Size:", BATCH_SIZE)
print("DROPOUT:", DROPOUT)
print("EPS:", EPS)
print("WEIGHT_DECAY", WEIGHT_DECAY)


# %%--------------------------------------- Augmenting -------------------------------------------------------------




# from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return {'image': image}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return {'image': img}


### From https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


### transform.Normalize(torch.mean(

### Set Augmentations:

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(128),
        #transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}

# %% ----------------------------------- Helper Functions --------------------------------------------------------------


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    print("Saving checkpoint")
    if not is_best:
        f_path = checkpoint_dir + 'checkpoint.pt'
        torch.save(state, f_path)
    else:
        f_path = best_model_dir + 'checkpoint_best_model4.pt'
        torch.save(state, f_path)


def y_generator(num_images):
    ### num_images is batchsize
    return np.concatenate([['NORMAL']*num_images,['VIRUS']*num_images,['BACTERIA']*num_images])


#%% ---------------------------------------------------Data Prep -------------------------------------------------------
### Data loader class goes here - Jane

class ImageData(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, mode='train',target=None,encoding='label'):
        self.root_dir = root_dir
        self.transform = transform
        self.images_f = [f for f in os.listdir(self.root_dir) if os.path.splitext(f)[1] == '.jpeg']
        self.mode = mode
        assert target is not None, "Pass a target type; NORMAL, VIRAL or BACTERIAL"
        self.labels_dic = {"red blood cell":0, "difficult":1, "gametocyte":2, "trophozoite":3,
                          "ring":4, "schizont":5, "leukocyte":6}
        self.target = target

    def __len__(self):
        return len(self.images_f)

    def __getitem__(self, index):
        image_index = self.images_f[index]
        image_path = os.path.join(self.root_dir, image_index)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            if encoding == 'one-hot':
                label_index = self.labels_f[index]
                label_path = os.path.join(self.root_dir, label_index)
                label = self.get_label_one_hot(label_path)
            elif encoding == 'label':
                label = target_label_encoding()
                return image, label
            else:
                print("Encoding type not supported. Defaulting to label encoding")
                label = target_label_encoding()
                return image, label

        elif self.mode == 'test':
            return image

    def target_label_encoding():
        le = LabelEncoder()
        le = le.fit(["NORMAL","VIRAL","BACTERIAL"])
        label = le.transform(self.target)
        return label


    def get_label_one_hot(self, label_path):
        labels = open(label_path, 'r').read().split('\n')
        label_one_hot = np.zeros(7)
        for i in labels:
            label_one_hot[self.labels_dic[i]] = 1
        return label_one_hot


# %% -------------------------------------- CNN Class ------------------------------------------------------------------

### CURRENTLY UNUSED
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
        self.drop = nn.Dropout2d(DROPOUT)
        self.act = nn.LeakyReLU()
        #self.act = torch.nn.Sigmoid()

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
        #x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        #x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        #x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        #return self.linear2(x)


# %% -------------------------------------- Training Prep ----------------------------------------------------------
#model = CNN().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=EPS,weight_decay=WEIGHT_DECAY)
#criterion = nn.BCEWithLogitsLoss()


# %% -------------------------------------- Train ----------------------------------------------------------

def train_model(model, criterion, optimizer, scheduler=None, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()



                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                ### Save checkpoints here
                if (epoch + 1) % SAVE_EVERY_CHECKPOINT == 0:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    is_best = False
                    save_ckp(checkpoint, is_best, CHECKPOINT_DIR, model_dir)
            if phase == 'train':
                epoch_loss_train = running_loss / dataset_sizes[phase]
                epoch_acc_train = running_corrects.double() / dataset_sizes[phase]

            elif phase == 'val':
                epoch_loss_val = running_loss / dataset_sizes[phase]
                epoch_acc_val = running_corrects.double() / dataset_sizes[phase]



            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


pneumonia_dataset = datasets.ImageFolder(root='../chest_xray_200x200/train',transform=data_transforms['train'])
pneumonia_val_dataset = datasets.ImageFolder(root='../chest_xray_200x200/val',transform=data_transforms['val'])

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(pneumonia_dataset,batch_size = BATCH_SIZE, shuffle = True, num_workers=0)
dataloaders['val'] = torch.utils.data.DataLoader(pneumonia_val_dataset,batch_size = BATCH_SIZE, shuffle = True, num_workers=0)

dataset_sizes = {}
dataset_sizes['train'] = len(pneumonia_dataset)
dataset_sizes['val'] = len(pneumonia_val_dataset)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 3)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler = exp_lr_scheduler,num_epochs=50)

