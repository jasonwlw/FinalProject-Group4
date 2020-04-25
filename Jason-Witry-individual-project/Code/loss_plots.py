import numpy as np

import matplotlib.pyplot as plt
with open("ResNet_fc8.log") as f0:
    losses = f0.readlines()

epochs = []
train_loss = []
test_loss = []
train_acc = []
test_acc = []
for line in losses:
    if line.startswith('Epoch'):
        end = line.find('|') - 1
        epochs.append(int(line[6:end]))

        tofind = 'Train Loss'
        trainloss0 = line.find(tofind)
        trainloss0 += len(tofind) + 1
        line = line[trainloss0:]
        trainloss1 = line.find(',')
        train_loss.append(float(line[:trainloss1]))

        tofind = 'Train Acc'
        trainloss0 = line.find(tofind)
        trainloss0 += len(tofind) + 1
        line = line[trainloss0:]
        trainloss1 = line.find('-') - 1
        train_acc.append(float(line[:trainloss1]))

        tofind = 'Test Loss'
        trainloss0 = line.find(tofind)
        trainloss0 += len(tofind) + 1
        line = line[trainloss0:]
        trainloss1 = line.find(',')
        test_loss.append(float(line[:trainloss1]))
        
        tofind = 'Test Acc'
        trainloss0 = line.find(tofind)
        trainloss0 += len(tofind) + 1
        line = line[trainloss0:]
        test_acc.append(float(line))

plt.plot(epochs, train_loss, label="Train")
plt.plot(epochs, test_loss, label="Test")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.show()

plt.plot(epochs, train_acc, label="Train")
plt.plot(epochs, test_acc, label="Test")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
