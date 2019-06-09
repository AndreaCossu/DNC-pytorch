
'''
Created on Aug 14, 2018

@author: andrea
'''

import random
import torch

clip = 5.

def train(dnc, input, target, masks, criterion, optimizer, device):
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

    outputs = torch.empty_like(input, device=device, dtype=torch.float32)
    for i in range(num_vectors):
        out, hidden_state, mem_state = dnc(input[:,i,:], hidden_state, mem_state)
        outputs[:,i,:] = out

    loss = criterion(outputs,target, masks)

    loss.backward()

    #torch.nn.utils.clip_grad_norm_(dnc.parameters(), clip)

    optimizer.step()

    return loss.item()

def train2(dnc, input, target, criterion, optimizer, device):
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

    return loss.item()

def get_dataset(vector_size, min_s, max_s, num_batches, device):
    '''

    Data matrix of size (B, L, V)
    Masks (L) represents the part of the target which have to be considered in the loss

    '''

    sequence_length = random.randint(min_s, max_s) # how many vectors in each input sequence
    total_length = 2 * sequence_length + 2 # extended number of vectors (input-target + 2 markers)

    shape = (num_batches, total_length, vector_size)
    inp_sequence = torch.zeros(shape, dtype=torch.float32, device=device)
    out_sequence = torch.zeros(shape, dtype=torch.float32, device=device)
    masks = torch.zeros(total_length, dtype=torch.float32, device=device)

    copy_vectors = torch.randint(0,2, (num_batches, sequence_length, vector_size - 1), device=device)
    # keep last column always 0 to place the marker that has value 1
    inp_sequence[:, :sequence_length, :-1] = copy_vectors
    out_sequence[:, sequence_length + 1:2 * sequence_length + 1, :-1] = copy_vectors

    inp_sequence[:, sequence_length, -1] = 1  # marker vector, end of input

    masks[sequence_length+1:2*sequence_length+1] = 1

    return inp_sequence, out_sequence, masks

def masked_BCE_with_logits(out, target, masks):
    '''
    Masked binary cross entropy loss with logits

    :param out - Output of model (B, L, V)
    :param target - (B, L, V)
    :param mask - (L) 1 only if corresponding vector in the sequence is relevant, 0 otherwise

    :return loss averaged over batch dimension
    '''


    first_normalizer = float(out.size(2) - 1) # vector_len

    out_masked = out[:, :, :-1]
    target_masked = target[:, :, :-1]

    # cross entropy
    L = -1 * torch.sum((target_masked * torch.log(out_masked) + (1-target_masked) * torch.log(1-out_masked)), dim=2) / first_normalizer

    '''
    second_normalizer = torch.sum(masks).float()

    # mask out irrelevant vector loss in the sequence
    masked_loss = torch.mv(L, masks) / second_normalizer # size (B)


    # mean over batches
    return torch.mean(masked_loss)
    '''

    return torch.mean(L)






def get_dataset2(vector_size, min_s, max_s, num_batches, device):
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
