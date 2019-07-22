'''
Created on Jul 26, 2018

@author: andrea
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.Controllers.ControllerLSTM import ControllerLSTM
from core.Controllers.ControllerMLP import ControllerMLP

from core.Memory import Memory


class DNC(nn.Module):
    '''
    DNC is the main module that encapsulate both Controller and Memory.
    '''


    def __init__(self,inputDim, hiddenDim, num_layers, outputDim, N, W, R, batch_size,
                 device, controller_type='LSTM', output_f=None):
        '''
        :param inputDim: input dimension
        :param hiddenDim: controller hidden dimension
        :param num_layers: number of hidden layers of controller
        :param outputDim: output dimension
        :param N: number of rows in memory
        :param W: number of columns in memory
        :param R: number of read heads
        :param batch_size: batch size
        :param device: gpu or cuda
        :param controller_type: LSTM or MLP
        :param output_f: optional function to be applied to output
        '''

        super(DNC,self).__init__()

        self.device = device
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.num_layers = num_layers
        self.outputDim = outputDim
        self.N = N
        self.W = W  # memory (N x W)
        self.num_read_heads = R
        self.batch_size = batch_size
        self.output_f = output_f

        # create controller
        if controller_type == 'LSTM':
            self.controller = ControllerLSTM(self.inputDim, self.hiddenDim, self.num_layers, self.outputDim, self.W, self.num_read_heads, self.device)
        elif controller_type == 'MLP':
            self.controller = ControllerMLP(self.inputDim, self.hiddenDim, self.num_layers, self.outputDim, self.W, self.num_read_heads, self.device)
        else:
            raise NameError("Error: controller must be LSTM or MLP.")

        # create memory
        self.memory = Memory(self.N, self.W, self.num_read_heads, self.batch_size, self.device)

        # linear layer to produce output
        self.read_l  = nn.Linear(self.num_read_heads * self.W, self.outputDim, bias=False).to(self.device)

        # DNC memory parameters with sizes
        self.dimensions = {
            'read_keys' : torch.Size([self.batch_size, self.W, self.num_read_heads]),
            'read_strengths' : torch.Size([self.batch_size, self.num_read_heads]),
            'write_key' : torch.Size([self.batch_size, self.W]),
            'write_strength' : torch.Size([self.batch_size]),
            'erase_vector' : torch.Size([self.batch_size, self.W]),
            'write_vector' : torch.Size([self.batch_size, self.W]),
            'free_gates' : torch.Size([self.batch_size, self.num_read_heads]),
            'allocation_gate' : torch.Size([self.batch_size]),
            'write_gate' : torch.Size([self.batch_size]),
            'read_modes' : torch.Size([self.batch_size, 3, self.num_read_heads]) # R vectors of dimension 3
            }

    def decompose_xi(self, full_xi):
        '''
        Decompose the xi, output of controller, in memory parameters
        '''

        xi = {}
        prevEnd = 0
        currEnd = 0
        for key in self.dimensions.keys():
            currEnd += self.reduce_dimensions(self.dimensions[key])
            xi[key] = full_xi[:, prevEnd:currEnd].clone().view(self.dimensions[key])  # clone propagates gradient
            prevEnd = currEnd

        # adapt domain of some memory parameters
        xi['read_strengths'] = self.oneplus(xi['read_strengths'])
        xi['write_strength'] = self.oneplus(xi['write_strength'])
        xi['erase_vector'] = torch.sigmoid(xi['erase_vector'])
        xi['free_gates'] = torch.sigmoid(xi['free_gates'])
        xi['allocation_gate'] = torch.sigmoid(xi['allocation_gate'])
        xi['write_gate'] = torch.sigmoid(xi['write_gate'])
        xi['read_modes'] = F.softmax(xi['read_modes'],dim=1)  # softmax on each of the R columns (3 elements per column)

        return xi

    def forward(self,input, hidden_state, memory_state):
        '''

        Use the controller and returns output and hidden state w.r.t. current input.
        Internally update memory parameters.

        :param input (B, V) where V is the length of the pattern vector
        :param hidden_state: tuple of ( (B, H), (B, H) ) or None if MLPController
        :param memory_state: list of memory state (the first element are read vectors)

        :return y_t (B, O) where O is the output dimension
        :return hidden_state: tuple of ( (B, H), (B, H) ) the next hidden state
        :return memory_state: updated memory state
        '''


        # take previous read vectors
        read_vectors = memory_state[0]

        vu_t, full_xi, hidden_state = self.controller(input, hidden_state, read_vectors.transpose(1,2).contiguous().view(self.batch_size, -1))

        xi = self.decompose_xi(full_xi)

        # update memory state
        read_vectors, memory_state = self.memory(xi['erase_vector'], xi['free_gates'], xi['allocation_gate'], xi['write_gate'], xi['read_modes'],
                xi['read_strengths'], xi['read_keys'], xi['write_vector'], xi['write_key'], xi['write_strength'], memory_state)


        read_adaptive = self.read_l(read_vectors.transpose(1,2).contiguous().view(self.batch_size, -1))

        # produce output
        y_t = torch.add(vu_t, read_adaptive)

        if self.output_f is not None:
            y_t = self.output_f(y_t)

        return y_t, hidden_state, [read_vectors] + memory_state


    def reset(self):
        '''
        Reset controller hidden state

        :return hidden_state: tuple of ( (B, H), (B, H) ) or None if MLPController
        '''

        memory_state = self.memory.reset()
        return self.controller.reset_hidden_state(self.batch_size), memory_state


    def update_batch_size(self,batch_size):
        '''
        Dynamically update DNC and Memory to take different batch size

        :param batch size: int

        '''

        self.batch_size = batch_size
        self.memory.batch_size = batch_size

        self.dimensions = {
            'read_keys' : torch.Size([self.batch_size, self.W, self.num_read_heads]),
            'read_strengths' : torch.Size([self.batch_size, self.num_read_heads]),
            'write_key' : torch.Size([self.batch_size, self.W]),
            'write_strength' : torch.Size([self.batch_size]),
            'erase_vector' : torch.Size([self.batch_size, self.W]),
            'write_vector' : torch.Size([self.batch_size, self.W]),
            'free_gates' : torch.Size([self.batch_size, self.num_read_heads]),
            'allocation_gate' : torch.Size([self.batch_size]),
            'write_gate' : torch.Size([self.batch_size]),
            'read_modes' : torch.Size([self.batch_size, 3, self.num_read_heads]) # R vectors of dimension 3
        }


    def oneplus(self, x):
        '''
        Each element of the output is in the range [1,+inf)
        '''

        return 1 + torch.log(1 + torch.exp(x))

    def reduce_dimensions(self, size):
        '''
        Compute total number of element in a tensor with Size size
        without including batch dimension
        '''

        if len(size) == 1:
            return 1
        else:
            return torch.prod(torch.tensor(size[1:])).item()


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def clip_tensor(self, tensor):
        '''
        Prevent nan when doing cross entropy loss
        '''

        return torch.clamp(tensor, min=-10., max=10.)
