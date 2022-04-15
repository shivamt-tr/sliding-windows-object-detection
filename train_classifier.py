# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 02:46:11 2022

@author: tripa
"""

# Import common python libraries
import os
import time
import copy
import random
import numpy as np
from PIL import Image
from utils import prepare_patchwise_data, majority_class_undersampling, load_transformed_batch

# Import PyTorch packages
import torch
from torch import optim
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter

# %%

# Root directory for VOC data
voc_root = os.path.join(os.getcwd(), 'VOC2007')
data_dir = os.path.join(os.getcwd(), 'data')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor', 'bg']

# Set these flags to true if data preparation and undersampling (to correct class imbalance) is required
prepare_data = False
undersample = False
image_sets = ['train', 'val']  # valid_image_sets = ["train", "trainval", "val", "test"]

if prepare_data and undersample:
    for image_set in image_sets:
        prepare_patchwise_data(voc_root, image_set)
        majority_class_undersampling(data_dir, object_categories, image_set, sampling_number=200)

# %%

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

train_dir = os.path.join(data_dir, 'train_balanced')
val_dir = os.path.join(data_dir, 'val_balanced')

train_files = os.listdir(train_dir)
val_files = os.listdir(val_dir)

# Shuffle file names
random.shuffle(train_files)
random.shuffle(val_files)

# 1000 files each
# train_files = train_files[:33]
# val_files = val_files[:33]

# Number of classes in the dataset
n_classes = 21

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
n_epochs = 15

# Input size for ResNet-50
input_size = 224

# Transformations to be applied on data
train_transforms = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

val_transforms = transforms.Compose([transforms.Resize(input_size),
                                    transforms.CenterCrop(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

# %%

# Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)

# Change number of softmax outputs
model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
num_ftrs = model.fc.in_features

# Last layer will output 21 softmax outputs, where 20 of them belong to original class and one belongs to background class
model.fc = torch.nn.Linear(num_ftrs, 21)
model.to(device)

# Set-up optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=1.5e-4, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

# Set-up loss function
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
# m = torch.nn.Sigmoid()  # Use if required over softmax outputs

val_acc_history = []
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Calculate the number of batches
n_train_batches = int(len(train_files)/batch_size)
n_val_batches = int(len(val_files)/batch_size)

for epoch in range(n_epochs):
    
    # Variables to record time taken in each epoch
    since = time.time()
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    # Each epoch will have a train and a test phase
    for phase in ['train', 'val']:
        
        if phase == 'train':
            model.train()  # Set model to training mode
            n_batches = n_train_batches
        else:
            model.eval()   # Set model to evaluate mode
            n_batches = n_val_batches

        running_loss = 0.0
        running_corrects = 0
        
        # Iterate over all the batches
        for j in range(n_batches):
            
            print('Batch {}/{}'.format(j+1, n_batches+1))
            
            if phase == 'train':
                # Get the train data and labels for the current batch
                batch_files = train_files[j*batch_size:(j+1)*batch_size]
                data, target = load_transformed_batch(train_dir, batch_files, train_transforms, object_categories)
            else:
                # Get the train data and labels for the current batch
                batch_files = val_files[j*batch_size:(j+1)*batch_size]
                data, target = load_transformed_batch(val_dir, batch_files, val_transforms, object_categories)
            
            data, target = data.to(device), target.to(device)
            
            # Make gradients zero before forward pass
            optimizer.zero_grad()
        
            # Set to True in training phase
            with torch.set_grad_enabled(phase == 'train'):
            
                outputs = model(data)
                loss = criterion(outputs, target)
                
                # _, preds = torch.max(outputs, 1)
        
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(torch.argmax(outputs, 1) == torch.argmax(target, 1))
        
        epoch_loss = running_loss / (n_batches*batch_size)
        epoch_acc = running_corrects.double() / (n_batches*batch_size)
    
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if phase == 'val':
            val_acc_history.append(epoch_acc)
        
    time_elapsed = time.time() - since
    print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # The below statements are run after every epoch for saving the model
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save the best model
    torch.save(model.state_dict(), 'model_' + str(epoch) + '.pth')
    
# After the training, print best validation accuracy
print('Best val Acc: {:4f}'.format(best_acc))

# Return these values: model, val_acc_history