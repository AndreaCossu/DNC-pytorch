'''
Created on Jul 18, 2018

@author: andrea
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.Controllers.BaseController import BaseController

class ControllerLSTM(BaseController):
    def __init__(self, inputDim, hiddenDim, num_layers, outputDim, W, R, device):
        
        super(ControllerLSTM, self).__init__(inputDim, hiddenDim, num_layers, outputDim, W, R, device)
        
        # LSTM layers   
        self.lstm = nn.LSTM(self.input_size, self.hiddenDim, num_layers=self.num_layers).to(self.device)     
        
    def forward(self, input, hidden_state, read_vectors):
        '''
        :param (B, V) input tensor 
        :param hidden_state: tuple of ( (B, H), (B, H) )
        :param read_vectors: (B, W*R) concatenation of read vectors
        
        :return vu_t (B, O) where O is the output dimension
        :return full_xi (B, xi_size) 
        :return hidden_state: tuple of ( (B, H), (B, H) )
        '''
        
        batch_size = input.size(0)
                
        # ensure column input and hidden vector
        input = input.view(batch_size,-1)
        # concatenate input with read vectors in one column vector
        nn_input = self.concatenate_vectors(input, read_vectors)
        
        processed, hidden = self.lstm(nn_input.unsqueeze(0), hidden_state)
        
        processed = self.clip_tensor(processed)
        
        vu_t = self.y_l(processed.squeeze())
        full_xi_t = self.xi_l(processed.squeeze())
        
        return self.clip_tensor(vu_t), self.clip_tensor(full_xi_t), hidden
    
        
    def reset_hidden_state(self, batch_size):
        # hidden is composed by hidden and cell state vectors
        return (torch.zeros(self.num_layers, batch_size,self.hiddenDim, device=self.device, requires_grad=True),
                torch.zeros(self.num_layers, batch_size,self.hiddenDim, device=self.device, requires_grad=True)
                )
        
        
        