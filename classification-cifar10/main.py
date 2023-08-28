# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:45:57 2022

@author: tripa
"""

import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import imshow, CIFAR10Dataset, confusion_matrix
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

network = 'CustomLeNet'  # Specify one of 'CustomCNN', 'SimpleNet', or 'CustomLeNet'
n_epochs = 10
learning_rate = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs('./data', exist_ok=True)

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = CIFAR10Dataset(root='./data', train=True,
                          download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=0)  # num_workers set to 0 in windows

testset = CIFAR10Dataset(root='./data', train=False,
                         download=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=0)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# %%

# Initialize convnet
if network == 'CustomCNN':
    from custom_cnn import CustomCNN
    net = CustomCNN()
if network == 'SimpleNet':
    from pytorch_cnn import SimpleNet
    net = SimpleNet()
if network == 'CustomLeNet':
    from pytorch_cnn import CustomLeNet
    net = CustomLeNet()

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
multistep_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,8], gamma=0.1)

for epoch in range(1, n_epochs+1):  # loop over the dataset multiple times

    print('Epoch {}/{}'.format(epoch, n_epochs))
    print('-' * 10)

    running_loss = 0.0

    for i, data in enumerate(tqdm(trainloader, desc='Batch'), start=0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
    
    multistep_scheduler.step()

    epoch_loss = running_loss / trainset.__len__()
    print('Loss: {:.3f}'.format(epoch_loss))    

print('Finished Training')
torch.save(net.state_dict(), 'cifar_net.pth')  # save model

# %%

# Test the model on a sample image

dataiter = iter(testloader)
images, labels = dataiter.next()
images = images.to(device)

# print images
imshow(make_grid(images.cpu()))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net.load_state_dict(torch.load('cifar_net.pth'))
net.to(device)

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

# %%

# Calculate accuracy on test dataset

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# %%

# Calculate accuracy of each class

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# store data labels and prediction labels of test dataset
data_labels = np.array((), dtype=int)
prediction_labels = np.array((), dtype=int)

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
        # append labels and predictions to the list
        data_labels = np.append(data_labels, labels.cpu().numpy())
        prediction_labels = np.append(prediction_labels, predictions.cpu().numpy())

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# plot confusion matrix
confusion_matrix(data_labels, prediction_labels, classes)