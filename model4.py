import numpy as np
import torch
import torch.nn as nn
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
SAVE_EVERY_CHECKPOINT = 20
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



def apply_random_transform(im0):
    y = im0[1]
    im0 = im0[0]
    ### if it has a minority class and no red blood cells, augment 15*, if minority class and red blood cell, augment it 7*

    ims = []
    ys = []
    im = copy.deepcopy(im0)
    pick_rot = np.random.randint(0,3)
    rotate = {0:cv2.ROTATE_90_CLOCKWISE,1:cv2.ROTATE_90_COUNTERCLOCKWISE,2:cv2.ROTATE_180}
    bright = np.arange(0.9,1.15,0.05)
    pick_bright = np.random.randint(0,len(bright))
    sharpness = np.arange(-2,2,0.25)
    pick_sharpness = np.random.randint(0,len(sharpness))
    flip = np.random.randint(0,2)
    pick_flip = np.random.randint(-1,2)
    noise_on = np.random.randint(0,4)
    ### Odd UMat src cv2 error
    if flip > 0:
        im = cv2.flip(im,pick_flip)
    im = cv2.rotate(im,rotate[pick_rot])
    im = Image.fromarray(im)
    im = ImageEnhance.Brightness(im)
    im = im.enhance(bright[pick_bright])
    im = ImageEnhance.Sharpness(im)
    im = im.enhance(sharpness[pick_sharpness])
    im = np.asarray(im)
    im = im / 255.
    if noise_on != 0:
        noise = np.random.normal(loc = 0, scale = 0.05, size = im.shape)
        im = np.clip(im + noise, 0, 1)
    ims.append(im)
        ys.append(y)
    return ims, ys



# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def read_data(datapath='../train/',test_size = 0.2,num_data = None,num_filetypes=1):
    im_list = []
    y_list = []
    bb_list = []
    num_files = 0
    dir = sorted(os.listdir(datapath))
    ys = datapath.split('/')[-1]
    for fil in dir:
        if fil.endswith('.png') or fil.endswith('.jpeg') or fil.endswith('.jpg'):
            im_list.append(datapath+fil)
            #y_list.append(ys)
        num_files += 1
        ### If there is target, image, and bounding box when this is 30 we will have 10 data points
        if num_data is not None:
            if num_files >= num_data*num_filetypes:
                break
        else:
            pass
    #im_list,y_list = align_data(im_list,y_list)
    return np.asarray(im_list)


def resize_im(im):
    desired_size = (200,200)
    return cv2.resize(im,desired_size)

def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x).squeeze()
        logits = logits.cpu()
        logits = torch.sigmoid(logits)
        pred_labels = logits > 0.5
        #pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    print("Saving checkpoint")
    if not is_best:
        f_path = checkpoint_dir + 'checkpoint.pt'
        torch.save(state, f_path)
    else:
        f_path = best_model_dir + 'checkpoint_best_model4.pt'
        torch.save(state, f_path)

def align_data(im_list):
    ### Align data and target based on the number in the filename
    pneu_type = [im.split('_')[2] for im in im_list]
    virus = []
    bacteria = []
    for i in range(len(pneu_type)):
        if pneu_type[i] == 'virus':
            virus.append(im_list[i])
        elif pneu_type[i] == 'bacteria':
            bacteria.append(im_list[i])
    return np.asarray(virus),np.asarray(bacteria)

def batch_generator(normal,virus,bacteria,num_images):
    norm = np.random.choice(normal,num_images,replace = False)
    vir = np.random.choice(virus,num_images,replace = False)
    bact = np.random.choice(bacteria,num_images,replace = False)
    return np.concatenate([norm, vir, bact])

def y_generator(num_images):
    return np.concatenate([['NORMAL']*num_images,['VIRUS']*num_images,['BACTERIA']*num_images])


def data_prep(start, BATCH_SIZE, normal, virus, bacteria, label_encoder):
    print("Reading/Preparing Data")
    ### SHOULD THIS GO IN TRAINING SECTION? PROBABLY
    batch = batch_generator(normal, virus, bacteria, int(BATCH_SIZE/3))
    y_train = y_generator(int(BATCH_SIZE/3))

    batch = np.asarray(list(map(cv2.imread,batch)))


    X_train = X_train.flatten()
    X_test = X_test.flatten()

    X_train = np.asarray(list(map(cv2.imread, X_train)))
    X_test = np.asarray(list(map(cv2.imread, X_test)))

    ### Move to augmentation file once it is ready
    X_train = np.asarray(list(map(resize_im, X_train)))
    X_test = np.asarray(list(map(resize_im, X_test)))

    X_train = np.asarray(list(map(image_cropping, zip(X_train, bb_train))))
    y_train = np.asarray(list(chain(*list(map(mlb.transform,X_train[:,1])))))
    X_train = np.asarray(list(chain(*X_train[:,0])))
    X_train = np.asarray(list(map(apply_random_transform,zip(X_train,y_train))))
    y_train = np.asarray(list(chain(*X_train[:,1])))
    X_train = np.asarray(list(chain(*X_train[:,0])))
    X_train, y_train = np.asarray([i for i in X_train if i is not None]), np.asarray([i for i in y_train if i is not None])
    
    #X_train = np.asarray(list(chain(*list(map(apply_random_transform,zip(X_train,y_train))))))

    X_test = np.asarray(list(map(image_cropping, zip(X_test, bb_test))))
    y_test = np.asarray(list(chain(*list(map(mlb.transform,X_test[:,1])))))
    X_test = np.asarray(list(chain(*X_test[:,0])))

    #X_train = X_train / 255.
    X_test = X_test / 255.

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.Tensor(y_train).to(device)
    y_test = torch.Tensor(y_test).to(device)

    X_train = X_train.view(len(X_train), 3, 200, 200)
    X_test = X_test.view(len(X_test), 3, 200, 200)

    return X_train, y_train, X_test, y_test



# %% -------------------------------------- CNN Class ------------------------------------------------------------------
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
        #self.act = torch.nn.Sigmoid()
        self.drop = nn.Dropout2d(0.1)
        #self.linear1 = nn.Linear(32*5*5, 400)  # input will be flattened to (n_examples, 32 * 5 * 5)
        #self.linear1_bn = nn.BatchNorm1d(400)
        #self.drop = nn.Dropout(DROPOUT)
        #self.linear2 = nn.Linear(400, 21)
        self.act = nn.LeakyReLU()

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

#%% ---------------------------------------------------Data Prep -------------------------------------------------------
# Make all images size of the minimum image size
X_path_full,y_full,bb = read_data('./train/',num_data=None)
y_full = list(map(target_read,y_full))
X_path_full, y_full = np.asarray(X_path_full), np.asarray(y_full)
### Encode multilabels
target_list = ["red blood cell", "difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"]
mlb = MultiLabelBinarizer(classes=target_list)
mlb.fit([target_list])
y_full = mlb.transform(y_full)

bb = list(map(open,bb))
bb = list(map(json.load,bb))

bb_lis = []
for im in bb:
    bb_lis.append(np.asarray(list(map(compute_centroid, im))))
bb = bb_lis


n_im = 8
start = 0
X_train, y_train, X_test, y_test = data_prep(start, n_im, X_path_full, y_full, bb, mlb)
bb = np.asarray(bb)
#x_train.requires_grad = True

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=EPS,weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()


# %% -------------------------------------- Train ----------------------------------------------------------
print("Starting training loop...")
model.train()
shuf = np.arange(len(X_path_full))

global_loss_test = 100
last_loss_test = 100
for epoch in range(N_EPOCHS):

    loss_train = 0
    #model.train()
    for batch in range(len(X_train)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(X_train[inds]).squeeze()
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test).squeeze()
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train/BATCH_SIZE, acc(X_train, y_train), loss_test, acc(X_test, y_test)))

    ### move model train here to save a model that can be resumed; it must be in training mode to resume
    if (epoch + 1) % SAVE_EVERY_MODEL == 0:
        print("Saving model...")
        torch.save(model.state_dict(), "./model4_augmented_cropped_epochs_tanh"+str(epoch)+".pt")

    model.train()

    if (epoch + 1) % SAVE_EVERY_CHECKPOINT == 0: 
        is_best = False
        if loss_test < last_loss_test:
            if loss_test < global_loss_test:
                global_loss_test = loss_test
                is_best = True

        last_loss_test = loss_test
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, is_best, CHECKPOINT_DIR, model_dir)

    ## Load new data
    start = n_im
    ### 128 images each time
    n_im += 8
    ### If we are at the end, shuffle the paths and reset
    if n_im >= len(X_path_full):
        np.random.shuffle(shuf)
        X_path_full = X_path_full[shuf]
        y_full = y_full[shuf]
        bb = bb[shuf]
        start = 0
        n_im = 8
    X_train, y_train, X_test, y_test = data_prep(start, n_im, X_path_full, y_full, bb, mlb)

