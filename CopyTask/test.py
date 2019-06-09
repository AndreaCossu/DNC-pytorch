'''
Created on Aug 28, 2018

@author: andrea
'''

import torch


def test(dnc, X, device, y=None, criterion=None, masks=None):

    with torch.no_grad():

        sequence_length = X.size(1)

        hidden_state, mem_state = dnc.reset()

        outputs = torch.empty_like(X, device=device)

        for i in range(sequence_length):
            output, hidden_state, mem_state = dnc(X[:,i,:], hidden_state, mem_state)
            outputs[:,i,:] = output

        if y is None:
            return outputs
        else:
            loss = criterion(outputs, y, masks)
            return outputs, loss.item()

def test2(dnc, X, device, y=None, criterion=None):

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


def accuracy(out, target, masks):
    with torch.no_grad():

        accuracy_unmasked = (out[:,:,:-1].round() == target[:,:,:-1]).float().sum(dim=2) / float(out.size(2) - 1) # vector_len

        accuracy = torch.mv(accuracy_unmasked, masks) / torch.sum(masks).float()

        acc_perc = torch.mean(accuracy).item()*100

        return acc_perc



def accuracy2(x, y):
    '''
    x, y: (batch_size, len_sequence, len_vector)

    Return the mean accuracy over batches.
    The accuracy is the percentage of bits correctly generated.
    '''

    with torch.no_grad():
        total_elements = torch.tensor(x.size(1) * x.size(2), dtype=torch.float32)

        accuracy = (torch.sigmoid(x).round() == y).float().sum(dim=1).sum(dim=1) / total_elements

        return torch.mean(accuracy).item()
