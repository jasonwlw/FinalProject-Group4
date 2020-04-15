import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import cv2
import os


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
    desired_size = (200,200)
    return cv2.resize(im,desired_size)

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

# %% ------------------------------- Read Data -------------------------------------------------------------------
### enter the paths to the data on your system
filepaths = ['../chest_xray/train/NORMAL/','../chest_xray/train/PNEUMONIA/']
for filepath in filepaths:
    X = read_data(datapath=filepath, num_data=None)
    if filepath.split('/')[-2] == 'PNEUMONIA':
        virus,bacteria = align_data(X)
    else:
        normal = X
        X = None

### dict for easy access to each array
all_types = {'normal':normal,'virus':virus,'bacteria':bacteria}

### Loop through dictionary keys
for types in all_types:
    ### Loop through file paths in the array
    for filepath in all_types[types]:
        im = cv2.imread(filepath)
        im = resize_im(im)
        ### numpy is binary file can save space/maybe load time
        #np.save(filepath.replace('chest_xray','chest_xray_200x200').replace('.jpeg',''),im)
        savepath = filepath.replace('chest_xray','chest_xray_200x200')
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


