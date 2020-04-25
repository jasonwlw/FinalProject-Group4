# Pneumonia-Detection
Using CNN's to automate bacterial or viral pneumonia detection

Data link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

after downloading the data, put the Group4 repository on the same level in the directory structure as the data; i.e. when listing the folder the repository and data are in, it should appear as:

'chest_xray'   'Group4-FinalProject'

This is to make it easier for the data_reorganizer to access your data and make the correct adjustments. Next, run the data reorganizer (Note: it looks for the data at '../chest_xray/'). If you check the directory above the github repository, you should have another folder named 'chest_xray_256x56' that has a now balanced validation set in it.

Now, you can train the model by running Final_Training_Script_group4.py. This will load the ResNet model and begin training for 25 epochs. To get validation plots, write the model out to a log, i.e.

python Final_Training_Script_group4.py > resnet18.log

and then use the loss_plots.py file.
