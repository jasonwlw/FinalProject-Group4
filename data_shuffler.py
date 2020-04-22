import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import cv2
import os

np.random.seed(42)
# %%-----------------------------------------------------Helper Functions ---------------------------------------

def read_data(datapath='../train/',test_size = 0.2,num_data = None,num_filetypes=1):
    im_list = []
    num_files = 0
    dir = sorted(os.listdir(datapath))
    for fil in dir:
        if fil.endswith('.png') or fil.endswith('.jpeg') or fil.endswith('.jpg'):
            im_list.append(datapath+fil)
        num_files += 1
        ### num_data = None gives all of the images
        if num_data is not None:
            ### How many file types are in the folder? If there are 2, i.e. .png and .txt,
            ### We will need to go through 20 files to get 10 images and 10 targets
            if num_files >= num_data*num_filetypes:
                break
        else:
            pass
    #im_list,y_list = align_data(im_list,y_list)
    return np.asarray(im_list)

def resize_im(im):
    desired_size = (1400,1400)
    return cv2.resize(im,desired_size)

def align_data(im_list):
    ### Find type of pneumonia from image path
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

# %% ------------------------------- Read Data -------------------------------------------------------------------
### enter the paths to the data on your system
filepaths_all = ['../chest_xray_256x256/train/NORMAL/','../chest_xray_256x256/train/PNEUMONIA/',
             '../chest_xray_256x256/val/NORMAL/','../chest_xray_256x256/val/PNEUMONIA/']

keys = ['train','val','test']
all_folders = {}
for key in keys:
    filepaths = [path for path in filepaths_all if key in path]
    all_types = {}
    for filepath in filepaths:
        X = read_data(datapath=filepath, num_data=None)
        if filepath.split('/')[-2] == 'PNEUMONIA':
            virus,bacteria = align_data(X)
            all_types['virus'] = virus
            all_types['bacteria'] = bacteria

        else:
            all_types['normal'] = X
            X = None
    all_folders[key] = all_types
### 5,863 images * 0.15 in validation gives ~ 830 images
### Nearest divisible by 3 is 831, gives 277 of each image
### There are 8 bacterial and 8 normal in validation
### Num bacteria and normal = 277 - 8 = 269
### num virus is 277
### Magic numbers will ensure this does not work if read_data does not read in more data than the numbers
### above
###

### change filepaths in val to train filepaths
for sets in all_folders['val']:
    all_folders['val'][sets] = np.asarray([i.replace('val','train') for i in all_folders['val'][sets]])

train_vir = np.random.choice(all_folders['train']['virus'],size=277,replace=False)
#train_vir = np.random.choice(all_folders['train']['virus'],size=2,replace=False)
### remove from train
all_folders['train']['virus'] = np.setdiff1d(all_folders['train']['virus'],train_vir)

train_bact = np.random.choice(all_folders['train']['bacteria'],size=277,replace=False)
#train_bact = np.random.choice(all_folders['train']['bacteria'],size=2,replace=False)
### remove from train
all_folders['train']['bacteria'] = np.setdiff1d(all_folders['train']['bacteria'],train_bact)

train_normal = np.random.choice(all_folders['train']['normal'],size=277,replace=False)
#train_normal = np.random.choice(all_folders['train']['normal'],size=2,replace=False)
### remove from train
all_folders['train']['normal'] = np.setdiff1d(all_folders['train']['normal'],train_normal)

for sets in all_folders['train']:
    all_folders['train'][sets] = np.append(all_folders['train'][sets],all_folders['val'][sets])


### put in validation
all_folders['val']['virus'] = np.append(all_folders['val']['virus'],train_vir)
### put in validation
all_folders['val']['bacteria'] = np.append(all_folders['val']['bacteria'],train_bact)
### put in validation
all_folders['val']['normal'] = np.append(all_folders['val']['normal'],train_normal)



### dict for easy access to each array

### Loop through dictionary keys
for folder in all_folders:
    all_types = all_folders[folder]
    for types in all_types:
    ### Loop through file paths in the array
        print("Folder",folder)
        print("Class",types)
        for filepath in all_types[types]:
            im = cv2.imread(filepath)
            im = resize_im(im)
            ### numpy is binary file can save space/maybe load time
            #np.save(filepath.replace('chest_xray','chest_xray_200x200').replace('.jpeg',''),im)
            savepath = filepath.replace('chest_xray','chest_xray_shuffled')
            if folder == 'val':
                savepath = savepath.replace('train','val')
            ### Alternatively, if your script is in the chest_xray folder:
            #savepath = filepath.replace('train','train_200x200')
            ### You may want to further edit the above for your system, or comment it out to overwrite the
            ### downloaded files.

            ### Quick and dirty edit to save all at once, make necessary directories.
            ### can comment out resizing if you want full images, or change the desired size in the resize im function.
            if types == 'normal':
                dirpath = '/'.join(savepath.split('/')[:-1])
                os.system('mkdir -p ' + str(dirpath))
                cv2.imwrite(savepath,im)
            elif types == 'virus':
                ### Replace PNEUMONIA directory with PNEUMONIA-VIRAL
                savepath = savepath.replace('PNEUMONIA','PNEUMONIA-VIRAL')
                dirpath = '/'.join(savepath.split('/')[:-1])
                os.system('mkdir -p ' + str(dirpath))
                cv2.imwrite(savepath, im)
            elif types == 'bacteria':
                savepath = savepath.replace('PNEUMONIA','PNEUMONIA-BACTERIAL')
                dirpath = '/'.join(savepath.split('/')[:-1])
                os.system('mkdir -p ' + dirpath)
                cv2.imwrite(savepath,im)
            else:
                print("Something went terribly wrong")


