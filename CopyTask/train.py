
'''
Created on Aug 14, 2018

@author: andrea
'''

import random
import torch

clip = 10.

def train(dnc, input, target, criterion, optimizer, device):
    '''
    dnc: core module
    input : single sequence
    target : expected output
    criterion : loss
    optimizer : optimization method
    device : cpu or cuda
    controller_type : LSTM or MLP
    '''

    num_vectors = input.size(1)

    optimizer.zero_grad()

    hidden_state, mem_state = dnc.reset()

    outputs = torch.empty_like(target, device=device, dtype=torch.float32)
    for i in range(num_vectors):
        out, hidden_state, mem_state = dnc(input[:,i,:], hidden_state, mem_state)

    for i in range(target.size(1)):
        out, hidden_state, mem_state = dnc(torch.zeros(input.size(0), input.size(2), device=device), hidden_state, mem_state)
        outputs[:, i, :]  = out

    loss = criterion(outputs,target)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(dnc.parameters(), clip)

    optimizer.step()

    return outputs, loss.item()


def get_dataset(vector_size, min_s, max_s, num_batches, device):
    '''
    Generate randomly batch_size sequences of len_sequence+1 vectors with len_vector items.

    :return dataset: the training data
    :return sequences: the target sequence to reconstruct
    '''

    sequence_length = random.randint(min_s, max_s) # how many vectors in each input sequence

    dataset = torch.zeros(num_batches, sequence_length+1, vector_size, device=device, dtype=torch.float32)
    sequences = torch.randint(0,2, (num_batches, sequence_length, vector_size-1), device=device, dtype=torch.float32)

    dataset[:,:-1,:vector_size-1] = sequences

    dataset[:,-1,-1] = 1. # end bit

    return dataset, sequences
