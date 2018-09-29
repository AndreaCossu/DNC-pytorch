'''
Created on Aug 28, 2018

@author: andrea
'''

import torch
import torch.nn as nn

class BaseController(nn.Module):
    def __init__(self, inputDim, hiddenDim, num_layers, outputDim, W, R, device):
        
        super(BaseController, self).__init__()
        
        self.device = device
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.num_layers = num_layers
        self.outputDim = outputDim
        self.W = W # columns of the memory (N x W)
        self.num_read_heads = R
        
        # some useful dimensions
        self.xi_size = (self.W * self.num_read_heads) + 3 * self.W + 5 * self.num_read_heads + 3
        self.input_size = self.inputDim + (self.num_read_heads * self.W )
        

        self.y_l = nn.Linear(self.hiddenDim, self.outputDim, bias=False).to(self.device)
        self.xi_l = nn.Linear(self.hiddenDim, self.xi_size, bias=False).to(self.device)

    def concatenate_vectors(self, x,y):
        '''
        Used to concatenate input vector and read vector and hidden vectors
        '''
        return torch.cat((x,y),1)
    
    def clip_tensor(self, tensor):
        '''
        Prevent nan when doing cross entropy loss
        '''
        
        return torch.clamp(tensor, min=-5.5, max=5.5)
    
    def forward(self, input, hidden_state, read_vectors):
        '''
        Abstract class, not implement forward method
        '''
        
        raise NotImplementedError()
    
    
        
        
        