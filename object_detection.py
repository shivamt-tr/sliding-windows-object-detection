# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 22:41:21 2022

@author: tripa
"""

# %% Run below cell when using colab

'''
# Mount google-drive
from google.colab import drive
drive.mount('/content/gdrive')

# Copy necessary files to the current environment
!cp gdrive/MyDrive/Colab/utils.py .
!cp gdrive/MyDrive/Colab/dataloader.py .
!cp gdrive/MyDrive/Colab/model_14.pth .

# Extract the VOC dataset in the current environment
!unzip gdrive/MyDrive/Colab/VOCdevkit.zip
'''

# %% 

import os
import numpy as np
from dataloader import VOCDataloader
from utils import sliding_windows, plot_bounding_boxes, non_maximum_suppression, mean_average_precision

import torch
from torchvision import transforms, models

# %%

# runtime = 'colab' or 'local'
runtime = 'local'

if runtime == 'local':
    res_dir = os.path.join(os.getcwd(), 'detection_results')
else:
    res_dir = os.path.join(os.getcwd(), 'gdrive', 'MyDrive', 'Colab', 'detection_results')
    
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# %%

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Number of classes in the dataset
n_classes = 21

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
n_epochs = 15

# Input size for ResNet-50
input_size = 224

# Load the ResNet-50 model
model = models.resnet50()

# Change number of softmax outputs
model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
num_ftrs = model.fc.in_features

# Last layer will output 21 softmax outputs, where 20 of them belong to original class and one belongs to background class
model.fc = torch.nn.Linear(num_ftrs, 21)
model.to(device)

# Load the trained model weights
if device.type == 'cpu':
    model.load_state_dict(torch.load('model_14.pth', map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load('model_14.pth'))

model.eval()  # Since we are using only for testing
model.requires_grad_(False)

# %%

image_set = 'test'

# Root directory for VOC data
voc_root = os.path.join(os.getcwd(), 'VOC2007')

# Create dataloader object
dataloader = VOCDataloader(voc_root, image_set=image_set)
print('Number of samples for', image_set, ':', dataloader.__len__())

test_transforms = transforms.Compose([# transforms.Resize(input_size),
                                      # transforms.CenterCrop(input_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

# %%

# instead of removing patches with confidence below background_thresh, I think now we should ignore the patches with output as 'background' class
stride = 20
window_size = 224
window_batch_size = 16
background_thresh = 0.8
iou_threshold = 0.1

sigmoid_transformation = torch.nn.Sigmoid()

gt_bboxes = []
pred_bboxes = []

# Create an empty numpy array to store the total list of predicted bounding-boxes (after applying nms)
total_final_boxes = np.empty((0, 7))

classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor', 'bg']

with torch.no_grad():
    
    for idx in range(4952):
        
        print('\nImage:', idx)
        
        # Get the data and target for the index
        data, target = dataloader.__getitem__(idx)
        
        # Save the data in the form of 3d-numpy array for plotting later
        # Convert PIL image to numpy array with 3 channels. https://stackoverflow.com/questions/44955656/how-to-convert-rgb-pil-image-to-numpy-array-with-3-channels
        original_data = np.array(data)
        
        # Apply transformations
        data = test_transforms(data)
        
        # Append each of the ground-truth bounding box in the current image to the
        # list in the following format [index, class_index, confidence_score, x1, y1, x2, y2]
        for obj in target['annotation']['object']:
            xmin = int(obj['bndbox']['xmin'])
            ymin = int(obj['bndbox']['ymin'])
            xmax = int(obj['bndbox']['xmax'])
            ymax = int(obj['bndbox']['ymax'])
            label = classes.index(obj['name'])
            gt_bboxes.append([idx, label, 100, xmin, ymin, xmax, ymax])
        
        # Convert data to numpy format and apply transpose to make channel dimensions as the last dimension
        image = data.cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = image.astype(float)
        
        # Create an empty numpy array to store the total list of detected bounding-boxes
        total_bounding_boxes = np.empty((0, 7))

        # Run sliding-window detection or the following aspect ratios
        for aspect_ratio in [1, 2, 3, 0.5, 0.33]:
        
            # Get the sliding windows accross the input image with given window-size, aspect_ratio and stride
            windows, bounding_boxes = sliding_windows(image, window_size=window_size, aspect_ratio=aspect_ratio, stride=stride)
    
            # print('Windows:', windows.shape, 'Bounding Boxes:', bounding_boxes.shape, 'Aspect Ratio:', aspect_ratio)
            
            # Calculate the number of batches based on the number of windows we got from sliding_windows()
            n_batches = int(len(windows)/window_batch_size)
            
            # Create an empty numpy array to store the sigmoid output vector for each window
            output = np.empty((0, 21))
            
            # Skip to the next iterations if no windows available for the current setting
            if n_batches == 0:
                continue
        
            # Iterate over the batches of windows
            for i in range(n_batches):
                
                # Get the windows for the current batch
                batch = windows[i*window_batch_size : (i+1)*window_batch_size]
                
                # Convert the batch of windows to float tensors
                batch_tensor = torch.from_numpy(batch)
                batch_tensor = batch_tensor.to(torch.float32)
    
                # Run the inputs through the model
                batch_tensor = batch_tensor.to(device)
                batch_output = model(batch_tensor)
        
                # Apply sigmoid on the softmax outputs
                batch_output = sigmoid_transformation(batch_output)
                
                # Convert the batch output tensors to numpy array
                batch_output = batch_output.cpu().detach().numpy()
                output = np.append(output, batch_output, axis=0)
        
            # Find the max confidence score and the index of that score
            max_values = np.max(output, axis=1)
            max_index = np.argmax(output, axis=1)
        
            # Concatenate the data sample index number, class label index, and the max confidence score column to bounding-boxes list
            bounding_boxes = np.c_[[idx for _ in range(len(max_index))], max_index, max_values, bounding_boxes[:len(output)]]
            
            # Keep the bounding-boxes where the max confidence score is above a certain-threshold
            bounding_boxes = bounding_boxes[bounding_boxes[:, 2] > background_thresh]
            
            # When the following line runs for the first time, assign the current bounding-boxes list to the total_bounding_boxes variable
            # print(total_bounding_boxes, len(total_bounding_boxes))
            if len(total_bounding_boxes) == 0:
                total_bounding_boxes = bounding_boxes
            # For the subsequent iterations, vertically stack the list of bounding-boxes to the total_bounding_boxes
            else:
                total_bounding_boxes = np.vstack((total_bounding_boxes, bounding_boxes))
            
            # print('Model output:', output.shape, 'Total Bounding Boxes:', total_bounding_boxes.shape)

        print('Total Bounding Boxes:')
        print('\tBefore applying NMS:', len(total_bounding_boxes))

        # Apply non-maximum suppression to get the final bounding boxes
        final_bb = non_maximum_suppression(total_bounding_boxes, iou_threshold=iou_threshold)
    
        print('\tAfter applying NMS:', len(final_bb))
        
        # Plot the final bounding boxes over the input image
        plot_bounding_boxes(original_data.copy(), final_bb, save=True, loc=res_dir)
        
        # When the following line runs for the first time, assign the current final bounding-boxes list to the total_final_boxes variable
        # print(total_final_boxes, len(total_final_boxes))
        if len(total_final_boxes) == 0:
            total_final_boxes = final_bb
        # For the subsequent iterations, vertically stack the list of bounding-boxes to the total_final_boxes
        else:
            total_final_boxes = np.vstack((total_final_boxes, final_bb))

        # Save the results after every 100 images and also save at the end of experiment
        if idx % 100 == 0 or idx == dataloader.__len__() - 1:
            np.save(os.path.join(res_dir, 'pred_results_'+str(idx)+'.npy'), np.array(total_final_boxes))
            np.save(os.path.join(res_dir, 'gt_results_'+str(idx)+'.npy'), np.array(gt_bboxes))

# %%

print('Mean Average Precision', mean_average_precision(gt_bboxes, gt_bboxes).item())