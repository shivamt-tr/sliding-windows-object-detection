# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:45:53 2022

@author: tripa
"""

import os.path
import pickle
import itertools
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class CIFAR10Dataset(Dataset):
    '''
    Custom Dataset Class for CIFAR10
    Acknowledgements: https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
    '''
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        
        base_folder = "cifar-10-batches-py"
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        
        train_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        test_list = ["test_batch"]
        
        if download:
            download_and_extract_archive(url, root, filename=filename)
        
        if train:
            downloaded_list = train_list
        else:
            downloaded_list = test_list
            
        self.data: Any = []
        self.targets = []
        
        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
        meta_path = os.path.join(root, base_folder, "batches.meta")
        with open(meta_path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data["label_names"]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self) -> int:
        return len(self.data)


def confusion_matrix(actual_labels, predicted_labels, label_names):
    
    # Create a 2D array to store confusion matrix entries
    result = np.zeros((len(label_names), len(label_names)))
    
    for i in range(len(actual_labels)):
        result[actual_labels[i]][predicted_labels[i]] += 1
    
    result_norm = result.astype('float') / result.sum(axis = 1)[: , np.newaxis] # Normalize our confusion matrix
    
    # Plot the confusion matrix
    figsize = (20, 20)
    n_classes = len(label_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(result, cmap = plt.cm.Blues)
    fig.colorbar(cax)

    # Label axes
    ax.set(title = 'Confusion Matrix',
           xlabel = 'Predicted Label',
           ylabel = 'True Label',
           xticks = np.arange(n_classes),
           yticks = np.arange(n_classes),
           xticklabels = label_names,
           yticklabels = label_names)
    
    # Set the xaxis labels to bottom 
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Adjust the label size 
    ax.yaxis.label.set_size(10)
    ax.xaxis.label.set_size(10)
    ax.title.set_size(10)

    # Set threshold for different colors 
    threshold = (result.max() + result.min()) / 2

    # Plot the text on each cell 
    for i, j in itertools.product(range(result.shape[0]) , range(result.shape[1])):
        plt.text(j, i ,f'{result[i , j]} ({result_norm[i , j]*100:.1f}%)',
                 horizontalalignment = 'center',
                 color = 'white' if result[i][j] > threshold else 'black',
                 size=10)