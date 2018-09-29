'''
Created on Aug 28, 2018

@author: andrea
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.Controllers.BaseController import BaseController

class ControllerMLP(BaseController):
    def __init__(self, inputDim, hiddenDim, num_layers, outputDim, W, R, device):
        super(ControllerMLP, self).__init__(inputDim, hiddenDim, num_layers, outputDim, W, R, device)

        self.l1 = nn.Linear(self.input_size, self.hiddenDim).to(self.device)        
        self.hidden_modules = nn.ModuleList([nn.Linear(self.hiddenDim, self.hiddenDim).to(self.device) for i in range(num_layers-1)])
                
    def forward(self, input, hidden_state, read_vectors):
        '''
        Use tanh as activation function as ReLu produces a lot of nan in the output.
        
        :param (B, V) input tensor 
        :param hidden_state: None
        :param read_vectors: (B, W*R) concatenation of read vectors
        
        :return vu_t (B, O) where O is the output dimension
        :return full_xi (B, xi_size) 
        :return hidden_state: None
        '''
        
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        x1 = self.concatenate_vectors(input, read_vectors)
        
        h = self.clip_tensor(torch.tanh(self.l1(x1)))
        
        for i in range(self.num_layers-1):
            h = self.clip_tensor(torch.tanh(self.hidden_modules[i](h)))
            
        vu_t = self.y_l(h)
        full_xi_t = self.xi_l(h)
        
        return self.clip_tensor(vu_t), self.clip_tensor(full_xi_t), hidden_state
    
    def reset_hidden_state(self, batch_size):
        return None
        