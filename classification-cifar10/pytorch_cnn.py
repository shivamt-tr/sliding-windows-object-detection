# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:28:52 2022

@author: tripa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomLeNet(nn.Module):
    '''
    Custom 8 layer convnet inspired by LeNet
    '''
    
    def __init__(self):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(3, 6, (3, 3), stride=1)
        self.avg_pool_1 = nn.AvgPool2d((2, 2), stride=1)
        
        self.conv_2 = nn.Conv2d(6, 12, (3, 3), stride=1)
        self.avg_pool_2 = nn.AvgPool2d((2, 2), stride=1)
        
        self.conv_3 = nn.Conv2d(12, 18, (3, 3), stride=1)
        self.avg_pool_3 = nn.AvgPool2d((2, 2), stride=1)
        
        self.conv_4 = nn.Conv2d(18, 24, (3, 3), stride=1)
        self.avg_pool_4 = nn.AvgPool2d((2, 2), stride=1)
        
        self.conv_5 = nn.Conv2d(24, 30, (3, 3), stride=1)
        self.avg_pool_5 = nn.AvgPool2d((2, 2), stride=1)
        
        self.conv_6 = nn.Conv2d(30, 36, (3, 3), stride=1)
        self.avg_pool_6 = nn.AvgPool2d((2, 2), stride=1)
        
        self.conv_7 = nn.Conv2d(36, 36, (3, 3), stride=1)
        self.avg_pool_7 = nn.AvgPool2d((2, 2), stride=2)
        
        self.conv_8 = nn.Conv2d(36, 120, (6, 6), stride=1)
        
        self.fully_connected_1 = nn.Linear(120, 84)
        self.fully_connected_2 = nn.Linear(84, 10)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        
        conv_1_output = self.tanh(self.conv_1(x))
        avg_pool_1_output = self.avg_pool_1(conv_1_output)

        conv_2_output = self.tanh(self.conv_2(avg_pool_1_output))
        avg_pool_2_output = self.avg_pool_2(conv_2_output)
        
        conv_3_output = self.tanh(self.conv_3(avg_pool_2_output))
        avg_pool_3_output = self.avg_pool_3(conv_3_output)
        
        conv_4_output = self.tanh(self.conv_4(avg_pool_3_output))
        avg_pool_4_output = self.avg_pool_4(conv_4_output)
        
        conv_5_output = self.tanh(self.conv_5(avg_pool_4_output))
        avg_pool_5_output = self.avg_pool_5(conv_5_output)
        
        conv_6_output = self.tanh(self.conv_6(avg_pool_5_output))
        avg_pool_6_output = self.avg_pool_6(conv_6_output)
        
        conv_7_output = self.tanh(self.conv_7(avg_pool_6_output))
        avg_pool_7_output = self.avg_pool_7(conv_7_output)
        
        conv_8_output = self.tanh(self.conv_8(avg_pool_7_output))  # The output here is of  shape (num_samples, 120, 1, 1)
        conv_8_output = conv_8_output.reshape(conv_8_output.shape[0], -1)  # We reshape it to make it flattened
        fc_1_output = self.tanh(self.fully_connected_1(conv_8_output))
        fc_2_output = self.fully_connected_2(fc_1_output)
        
        return fc_2_output
    