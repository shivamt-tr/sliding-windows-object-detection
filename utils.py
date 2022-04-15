# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 23:03:23 2022

@author: tripa
"""

import os
import cv2
import math
import torch
import random
import shutil
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from dataloader import VOCDataloader

def sliding_windows(image, window_size=100, aspect_ratio=1, stride=10):
    '''
    The sliding_windows function generates a numpy array of varying window sizes
    slided across the input image

    Parameters
    ----------
    image : 3-d numpy array
        representing an image (h, w, c)
    window_size : int, optional
        default window_size to be considered for sliding across the image
    aspect_ratio : float, optional
        aspect ratio for window size (could be 1, 2, 3, 0.5, 0.33, or any other number)
    stride : int, optional
        step size for sliding the window

    Returns
    -------
    windows : numpy array
        collection of 3-d numpy arrays representing the image window
    bounding_boxes : numpy array
        Each row in the numpy array represents a bounding-box [x1, y1, x2, y2], where
        (x1, y1) represents the top-left coordinates of the bounding-box
        (x2, y2) represents the bottom-right coordinates of the bounding-box
    '''

    # Calculate window_height and width for some given aspect_ratio
    window_height = int(window_size * math.sqrt(aspect_ratio))
    window_width = int(window_size / math.sqrt(aspect_ratio))
        
    # Create empty numpy array to store the windows
    windows = np.empty((0, 3, window_height, window_width))
    
    # Create empty numpy array to store the bounding-boxes for each window
    bounding_boxes = np.empty((0, 4))

    # Slide across the input image with a step-size of input stride
    for y in range(0, image.shape[0]-window_height, stride):
        for x in range(0, image.shape[1]-window_width, stride):
            
            # Extract the window
            window = image[y:y+window_height, x:x+window_width]
            
            # Transpose the window such that the channel dimension becomes the first dimension
            window = np.transpose(window, (2, 0, 1))
            
            # Append the window the windows array
            windows = np.append(windows, [window], axis=0)
            
            # Append the bounding-box for this window to the bounding_boxes array
            bounding_boxes = np.append(bounding_boxes, [[x, y, x+window_width, y+window_height]], axis=0)

    return windows, bounding_boxes

def plot_bounding_boxes(image, bounding_boxes, save=False, loc=None):
    '''
    Function to display the bounding-box rectangles in the input image
    along with class-name and confidence score
    
    Parameters
    ----------
    image : numpy array
    bounding_boxes : numpy array
        Each row in the numpy array represents a bounding-box [index, label, conf, x1, y1, x2, y2], where
        index represents the data sample index
        label represents the index of the class label (between 0-21)
        c represents the class objectness score
        (x1, y1) represents the top-left coordinates of the bounding-box
        (x2, y2) represents the bottom-right coordinates of the bounding-box
    '''
    
    classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor', 'bg']
    
    # Draw rectangle for each bounding-box in the bounding_boxes array
    for bb in bounding_boxes:
        # Extract the top-left coordinates and bottom-right coordinates of the bounding box
        x1, y1, x2, y2 = bb[3:].astype(int)
        # Extract the confidence score and class label
        class_index, confidence = int(bb[1]), bb[2]
        # Draw the rectangle over the image using the top-left coordinates and the bottom-right coordinates
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        # Write class label
        cv2.putText(image, classes[class_index], (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)
        # Write confidence score
        cv2.putText(image, str("{:.2f}".format(confidence)), (x1+5,y1+45), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)
        
    # Save and display the resulting image
    plt.imshow(image)

    if save:
        if len(bounding_boxes) == 0:
            print('No object detected')
        else:
            file_name = classes[class_index] + '_' + str(int(bb[0])) + '.png'
            plt.savefig(os.path.join(loc, file_name), bbox_inches='tight')
    plt.show()

def intersection_over_union(bb1, bb2):
    '''
    Implementation of Intersection over Union.
    The IoU is calculated between the two bounding boxes.
    The format of bounding box is [x1, y1, x2, y2], where
    (x1, y1) represents the top-left coordinates of the bounding-box
    (x2, y2) represents the bottom-right coordinates of the bounding-box
    '''
    
    # Extract the top-left coordinates and bottom-right coordinates of the bounding-box
    box1_x1, box1_y1, box1_x2, box1_y2 = bb1
    box2_x1, box2_y1, box2_x2, box2_y2 = bb2
    
    # If there is no overlap between the bounding boxes then intersection is zero
    if box1_x2 < box2_x1 or box2_x2 < box1_x1 or box1_y2 < box2_y1 or box2_y2 < box1_y1:
        intersection = 0
    # If there is an overlap, then calculate the dimensions (height and width)
    # of the overlapped region and find the area
    else:
        width = box2_x2 - box1_x1 if box1_x1 > box2_x1 else box1_x2 - box2_x1
        height = box2_y2 - box1_y1 if box1_y1 > box2_y1 else box1_y2 - box2_y1
        intersection = width * height
    
    # Calculate union using the set-union formula
    union = (box1_x2-box1_x1)*(box1_y2-box1_y1) + (box2_x2-box2_x1)*(box2_y2-box2_y1) - intersection
    
    # Return intersection over union
    return intersection/union

def non_maximum_suppression(bounding_boxes, iou_threshold=0.6):
    '''
    Function to apply non-maximum-suppression over the input bounding boxes
    
    Parameters
    ----------
    bounding_boxes : numpy array
        Each row in the numpy array represents a bounding-box [index, label, conf, x1, y1, x2, y2], where
        index represents the data sample index
        label represents the index of the class label (between 0-21)
        c represents the class objectness score
        (x1, y1) represents the top-left coordinates of the bounding-box
        (x2, y2) represents the bottom-right coordinates of the bounding-box
    iou_threshold : float, optional
        Bounding boxe having IoU with max-confidence boundind-box greater than
        iou_threshold are discarded.

    Returns
    -------
    numpy array of remaining bounding boxes after applying non-maximum-suppression 
    '''
    
    # Create an array to store the final list of bounding boxes
    final_bb = np.empty((0, bounding_boxes.shape[1]))
    
    # Run till there is at least one bounding box left
    while(len(bounding_boxes)):
        
        # print('Bounding Boxes Left:', len(bounding_boxes))
        
        # Find the bounding box with highest confidence score
        idx_max_confidence = np.argmax(bounding_boxes[:, 2])
        max_confidence_bb = bounding_boxes[idx_max_confidence]
        
        # Add the highest confidence bounding box to the final array of selected bounding-boxes
        final_bb = np.append(final_bb, [max_confidence_bb], axis=0)
        
        # Remove the max confidence bounding box from the array
        bounding_boxes = np.delete(bounding_boxes, idx_max_confidence, axis=0)
        
        # For the remaininng bounding boxes, delete the one's having IoU (with max confidence bounding box) greater than threshold
        index_to_delete = list()
        for i, bb in enumerate(bounding_boxes):
            if(intersection_over_union(max_confidence_bb[3:], bb[3:]) > iou_threshold):
                index_to_delete.append(i)
        # print('To delete', index_to_delete)
        bounding_boxes = np.delete(bounding_boxes, index_to_delete, axis=0)
    
    # print('Final bounding-boxes', final_bb)
    return final_bb

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=21, save=True, loc=None):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor', 'bg']

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                
                iou = intersection_over_union(detection[3:], gt[3:])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precision = torch.trapz(precisions, recalls).item()
        average_precisions.append(average_precision)
        
        print('Class: {}, Average Precision: {:.4f}'.format(classes[c], average_precision))
    
        # Plot P-R Curve
        plt.title('Precision-Recall for Class:'+classes[c])
        # plt.text(20, 20, 'Average Precision ='+str(average_precision), fontsize=22)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recalls.to('cpu').numpy(), precisions.to('cpu').numpy(), color='red', linewidth=1)
        plt.savefig(os.path.join(loc, classes[c]), bbox_inches='tight')
        plt.show()
    
    return sum(average_precisions) / len(average_precisions)

def one_hot_encode(data, n_classes):
    '''
    Function to one-hot-encode class labels given input data and number of classes.
    Parameters
    ----------
    data : np.array
        numerical data denoting the class of input
    n_classes: int
        number of classes in the dataset
    Returns
    -------
    encoded_labels: np.array
        one_hot_encoded labels
    Example
    -------
    If the input is [0, 2, 3, 1], the one_hot_encoded output will be
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
    '''
    
    # Create an array of zeros to store the one-hot-encoded labels
    encoded_labels = np.zeros((len(data), n_classes))
    # Enumerate over each data file and one-hot-encode labels of each sample
    for i, x in enumerate(data):
        encoded_labels[i][x] = 1
    
    return encoded_labels

def load_transformed_batch(data_dir, batch_files, data_transforms, object_categories):
    '''
    This function reads a list of batch_files from the given data_dir, apply some transformations on it,
    and return the data and labels of type tensor suitable for input to PyTorch models.
    Parameters
    ----------
    data_dir : path
        path of folder containing the training/testing images.
    batch_files : list
        list of file names to be considered for loading/transforming.
    data_transforms : transforms
        PyTorch pipeline for applying transformations on the data.
    object_categories : list
        list containing all the class labels in proper indexed order.

    Returns
    -------
    data : tensor
        tensor containing transformed data corresponding to the list of input files.
    labels : tensor
        tensor containing corresponding labels.
    '''
    
    labels = []
    data = []
    n_classes = len(object_categories)
    
    # Enumerate over all the files in the batch_files list
    for i, x in enumerate(batch_files):
        
        # Open image as PIL and apply transformations
        image = Image.open(os.path.join(data_dir, x)).convert("RGB")
        image = data_transforms(image)
        
        # For the first iteration create a tensor
        if i == 0:
            data = image.reshape(1, 3, 224, 224)
        # For subsequent iterations concatenate tensors to the data
        else:
            data = torch.cat([data, image.reshape(1, 3, 224, 224)], axis=0)
        
        # Get the image label using file name, i.e. file names are of format 'labelname_index.jpg'
        name = x.split('_')[0]
        # Get the index of this label
        label = object_categories.index(name)
        # Add the label to the list
        labels.append(label)
    
    # One-hot-encode labels and convert to tensor
    labels = one_hot_encode(labels, n_classes)
    labels = torch.from_numpy(labels)
    
    return data, labels


def majority_class_undersampling(data_dir, object_categories, image_set='train', sampling_number=200):

    # Get the list of files in the data_dir
    data_files = os.listdir(os.path.join(data_dir, image_set))
    
    # Destination folder to save balanced data
    data_dest = os.path.join(data_dir, image_set+'_balanced')
    
    print('Number of samples per class post undersampling majority classes:')
    
    # For each category in object_categories
    for cat in object_categories:
        
        # Get the list of files corresponding to this object category
        cat_files = [x for x in data_files if x.startswith(cat)]
    
        # If number of examples of this class is greater than 'sampling_number' then randomly sample 'sampling_number' files
        if len(cat_files) > sampling_number:
            cat_files = random.sample(cat_files, sampling_number)

        # Copy the final list of sampled files (or original list of files in case sampling not required) to destination directory
        for file in cat_files:
            shutil.copy(os.path.join(data_dir, image_set, file), data_dest)
            
        # Print category label and number of samples for this category
        print(cat, len(cat_files))
        
def prepare_patchwise_data(voc_root, image_set='train'):
    
    # Valid image_sets in the PascalVOC data
    # valid_image_sets = ["train", "trainval", "val", "test"]
    
    count = 0  # Count variable to keep track of number of preprocessed images saved
    background_search_iter = 1  # Number of times to extract random patches from image
    random_patch_size = 56  # Size of random patch to be selected from image (roughly 1/4th of ResNet50 input size, i.e. 224/4)
    minimum_patch_size = 10  # Minimum size of random patch to be selected from the image
    background_iou_threshold = 0.3  # IoU threshold to assign a random patch as a background
    
    # Create dataloader object for this image_set
    dataloader = VOCDataloader(voc_root, image_set=image_set)
    print('Number of samples for', image_set, ':', dataloader.__len__())
    
    # For each image present in the data of this image_set
    for i in range(dataloader.__len__()):
        
        # Get the image and target for index
        image, target = dataloader.__getitem__(i)
        
        # For each object in the input image
        for obj in target['annotation']['object']:
            
            # Extract label name and bounding box coordinates for this image
            label = obj['name']
            xmin = int(obj['bndbox']['xmin'])
            ymin = int(obj['bndbox']['ymin'])
            xmax = int(obj['bndbox']['xmax'])
            ymax = int(obj['bndbox']['ymax'])
            
            # Convert PIL image to numpy array
            image_array = np.array(image)
            height, width, _ = image_array.shape
            
            # Extract the foreground from image array using ground truth bounding box coordinates
            foreground_image = image_array[ymin:ymax, xmin:xmax]
        
            # Resize the foreground to suitable size
            foreground_image = cv2.resize(foreground_image, (224, 224))
            
            # Save this foreground patch and update count variable
            Image.fromarray(foreground_image).save(os.path.join(os.getcwd(), 'data', image_set, label+'_'+str(count)+'.jpg'))
            count += 1
            
            # Extract random patches from image and save the patches with low IoU with GT, these are considered as background patches
            for _ in range(background_search_iter):
    
                # Select the top left corner for the random patch
                x, y = random.randint(minimum_patch_size, width-1-random_patch_size), random.randint(minimum_patch_size, height-1-random_patch_size)
                
                # Extract the random patch of size random_patch_size
                random_patch = image_array[y:y+random_patch_size, x:x+random_patch_size]
                
                # Calculate IoU of random patch with foreground (ground truth) object
                iou = intersection_over_union([x, y, random_patch_size, random_patch_size], [xmin, ymin, xmax-xmin+1, ymax-ymin+1])
                
                # If IoU is below threshold then consider it a background
                if iou < background_iou_threshold:
                    
                    # Resize the random patch to suitable size
                    background_image = cv2.resize(random_patch, (224, 224))
                    
                    # Save this random patch as an image of class background and update count variable
                    Image.fromarray(background_image).save(os.path.join(os.getcwd(), 'data', image_set, 'bg_'+str(count)+'.jpg'))
                    count += 1