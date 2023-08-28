# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:45:15 2022

@author: tripa
"""

import torch
import torch.nn as nn

class Conv2d():
    '''
    Custom Conv2d
    Note: Bias is not used
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize random kernels of given filter size
        self.kernel = torch.randn((out_channels, in_channels, kernel_size[0], kernel_size[0]), requires_grad=True)

    def __call__(self, x):
        
        n_samples = x.shape[0]
        input_size = x.shape[-1]

        # Initialize output tensor with zeros
        # Note: For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and 
        # stride ‘𝑠’ our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓) / 𝑠} + 1].
        output_dim = int((input_size-self.kernel_size[0])/self.stride)+1
        output = torch.zeros(n_samples, self.out_channels, output_dim, output_dim)
        
        for n in range(n_samples):
            for i in range(self.out_channels):
                for j in range(output_dim):
                    for k in range(output_dim):
                        # print(n, i, j, k, self.kernel[i].shape, x[n][..., j*self.stride:j*self.stride+self.kernel_size[0], k*self.stride:k*self.stride+self.kernel_size[0]].shape)
                        output[n][i][j][k] = torch.sum(self.kernel[i] * x[n][..., j*self.stride:j*self.stride+self.kernel_size[0], k*self.stride:k*self.stride+self.kernel_size[0]])
                
        return output

class AvgPool2d():
    '''
    Custom AvgPool2d
    '''
    
    def __init__(self, kernel_size, stride=1, padding=0):

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def __call__(self, x):
        
        n_samples = x.shape[0]
        n_channels = x.shape[1]
        input_size = x.shape[-1]

        # Initialize output tensor with zeros
        # Note: For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and 
        # stride ‘𝑠’ our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓) / 𝑠} + 1].
        output_dim = int((input_size-self.kernel_size[0])/self.stride)+1
        output = torch.zeros(n_samples, n_channels, output_dim, output_dim)
        
        for n in range(n_samples):
            for i in range(n_channels):
                for j in range(output_dim):
                    for k in range(output_dim):
                        output[n][i][j][k] = torch.mean(x[n][..., j*self.stride:j*self.stride+self.kernel_size[0], k*self.stride:k*self.stride+self.kernel_size[0]])
                
        return output


class Linear():
    '''
    Custom fully-connected layer
    '''
    
    def __init__(self, n_input, n_output):
        
        self.n_input = n_input
        self.n_output = n_output
        self.weights = torch.randn((self.n_input, self.n_output), requires_grad=True)
        self.biases = torch.randn((self.n_output), requires_grad=True)
    
    def __call__(self, x):
        
        # return the matrix multiplication of input 'x' and weights and added with biases
        return (x @ self.weights) + self.biases


class CustomCNN(nn.Module):
    
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # The first convolution layer consists of 6 filters of size (5, 5), 
        # Note: It takes three channel input image
        self.conv_1 = Conv2d(3, 6, (5, 5), stride=1)
        
        # The next layer is an average pool layer consisting of filters of size (2, 2) and stride=2
        # Note: This halves the size of input
        self.avg_pool_1 = AvgPool2d((2, 2), stride=2)
        
        # The next layer is a convolution layer consisting of 16 filters of size (5, 5)
        self.conv_2 = Conv2d(6, 16, (5, 5))
        
        # The next layer is an average pool layer consisting of filters of size (2, 2) and stride=2
        # Note: This halves the size of input
        self.avg_pool_2 = AvgPool2d((2, 2), stride=2)
        
        # The next layer is a convolution layer consisting of 120 filters of size (5, 5)
        self.conv_3 = Conv2d(16, 120, (5, 5))

        # The last two layers are fully connected layers
        self.fully_connected_1 = Linear(120, 84)
        self.fully_connected_2 = Linear(84, 10)
        
        # Skipping below lines as I am using torch.tanh
        # The architecture uses Tanh activation function
        # self.tanh = nn.Tanh()
        
        # Trainable parameters
        self.parameters_list = [self.conv_1.kernel,
                                self.conv_2.kernel,
                                self.conv_3.kernel,
                                self.fully_connected_1.weights,
                                self.fully_connected_1.biases,
                                self.fully_connected_2.weights,
                                self.fully_connected_2.biases]
    
    def parameters(self):
        '''
        Returns trainable parameters
        '''
        for p in self.parameters_list:
            yield p
        
    def forward(self, x):
        
        conv_1_output = torch.tanh(self.conv_1(x))
        avg_pool_1_output = self.avg_pool_1(conv_1_output)
        conv_2_output = torch.tanh(self.conv_2(avg_pool_1_output))
        avg_pool_2_output = self.avg_pool_2(conv_2_output)
        conv_3_output = torch.tanh(self.conv_3(avg_pool_2_output))  # The output here is of shape (num_samples, 120, 1, 1)
        conv_3_output = conv_3_output.reshape(conv_3_output.shape[0], -1)  # We reshape it to make it flattened
        fc_1_output = torch.tanh(self.fully_connected_1(conv_3_output))
        fc_2_output = self.fully_connected_2(fc_1_output)
        
        return fc_2_output

# %%

###############################################################################
#              Test the Custom Conv2d and AvgPool2d Implementation            #
###############################################################################

if __name__ == "__main__":

    torch.manual_seed(0)
    
    x = torch.randn((1, 1, 7, 7))  # random input
    conv_custom = Conv2d(1, 1, (5, 5), stride=1) # custom conv2d
    conv_pytorch = torch.nn.Conv2d(1, 1, (5, 5), stride=1, bias = False)  # pytorch implementation
    conv_custom.kernel = conv_pytorch.weight  # make kernels same for both
    conv_diff = torch.mean(torch.abs(conv_custom(x) - conv_pytorch(x)))
    
    print('Mean of difference between the output from custom conv2d vs pytorch implementation: {:.6f}'.format(conv_diff.item()))
    
    avgpool_custom = AvgPool2d((5, 5), stride=1)  # custom avgpool2d
    avgpool_pytorch = torch.nn.AvgPool2d((5, 5), stride=1)  # pytorch implementation
    avgpool_diff = torch.mean(torch.abs(avgpool_custom(x) - avgpool_pytorch(x)))
    
    print('Mean of difference between the output from custom avgpool2d vs pytorch implementation: {:.6f}'.format(avgpool_diff.item()))
