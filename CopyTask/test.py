'''
Created on Aug 28, 2018

@author: andrea
'''

import torch

def test(dnc, X, device, y=None, criterion=None):

    with torch.no_grad():

        sequence_length = X.size(1)

        hidden_state, mem_state = dnc.reset()

        outputs = torch.empty(X.size(0), X.size(1)-1, X.size(2)-1, device=device)

        for i in range(sequence_length):
            output, hidden_state, mem_state = dnc(X[:,i,:], hidden_state, mem_state)

        for i in range(outputs.size(1)):
            out, hidden_state, mem_state = dnc(torch.zeros(X.size(0), X.size(2), device=device), hidden_state, mem_state)
            outputs[:, i, :]  = out

        if y is None:
            return outputs
        else:
            loss = criterion(outputs, y)
            return outputs, loss.item()



def accuracy(x, y):
    '''
    x, y: (batch_size, len_sequence, len_vector)

    Return the mean accuracy over batches.
    The accuracy is the percentage of bits correctly generated.
    '''

    with torch.no_grad():
        total_elements = torch.tensor(x.size(1) * x.size(2), dtype=torch.float32)

        accuracy = (torch.sigmoid(x).round() == y).float().sum(dim=1).sum(dim=1) / total_elements

        return torch.mean(accuracy).item()
