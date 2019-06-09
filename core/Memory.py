'''
Created on Jul 22, 2018

@author: andrea
'''

import torch
import torch.nn.functional as F
import torch.nn as nn

def batch_ger(x,y):
    '''
    Outer product preserving batch dimension

    :param x (B, X)
    :param y (B, Y)

    :return (B, X, Y) outer batch-product tensor
    '''

    return torch.bmm(x.unsqueeze(2), y.unsqueeze(2).transpose(1,2))


class TemporalMemoryLinkage(nn.Module):
    '''
    Implements Temporal Memory Linkage as a PyTorch Module.
    It is called by Memory during memory state computation
    '''

    def __init__(self, N, batch_size, device):
        super(TemporalMemoryLinkage, self).__init__()

        self.N = N
        self.batch_size = batch_size
        self.device = device

        self.reset()

    def reset(self):
        link_matrix = torch.zeros(self.batch_size, self.N, self.N, device=self.device, requires_grad=True)
        precedence_weighting = torch.zeros(self.batch_size, self.N, device=self.device, requires_grad=True)

        return link_matrix, precedence_weighting

    def forward(self, link_matrix, precedence_weighting, write_weighting, read_weightings):
        '''
        TODO: use sparse link matrix to improve computational efficiency

        Update Link Matrix (B, N, N)

        '''

        outer = batch_ger(write_weighting, precedence_weighting) # B, N, N

        outer_sum = write_weighting.unsqueeze(1) + write_weighting.unsqueeze(2) # B, N, N
        link_matrix = ( (1 - outer_sum) * link_matrix) + outer

        # Add this extra component to manage zero-diagonal automatically
        link_matrix = (1 - torch.eye(self.N, device=self.device)) * link_matrix

        precedence_weighting = self.update_precedence(precedence_weighting, write_weighting)

        forward, backward = self.update_forward_backward_weightings(link_matrix, read_weightings)

        return forward, backward, link_matrix, precedence_weighting

    def update_precedence(self, precedence_weighting, write_weighting):
        '''
        Update precedence weightings (B, N)
        '''

        precedence_weighting = ( (1 - torch.sum(write_weighting, dim=1)).view(-1,1) * precedence_weighting) + write_weighting

        return precedence_weighting

    def update_forward_backward_weightings(self, link_matrix, read_weightings):
        '''
        Update a (B, N, R) tensor of forward weightings.
        Update a (B, N, R) tensor of backward weightings)
        '''

        forward_weightings = torch.bmm(link_matrix, read_weightings)
        backward_weightings = torch.bmm(link_matrix.transpose(1,2), read_weightings)

        return forward_weightings, backward_weightings


class DynamicMemoryAllocation(nn.Module):
    '''
    Implements Dynamic Memory Allocation as a PyTorch Module.
    It is called by memory during memory state computation
    '''

    def __init__(self, N, batch_size, device):
        super(DynamicMemoryAllocation, self).__init__()

        self.N = N
        self.batch_size = batch_size
        self.device = device

        self.reset()

    def reset(self):
        memory_usage = torch.zeros(self.batch_size, self.N, device=self.device, requires_grad=True)

        return memory_usage


    def forward(self, memory_usage, free_gates, write_weighting, read_weightings):
        memory_usage = self.update_memory_usage(memory_usage, free_gates, write_weighting, read_weightings)

        sorted_indices, ordered_usage = self.update_free_list(memory_usage)

        allocation_weights = self.update_allocation_weights(sorted_indices, ordered_usage)

        return allocation_weights, memory_usage

    def update_memory_usage(self, memory_usage, free_gates, write_weighting, read_weightings):
        '''
        Update memory usage vector (B, N)

        :param (B, R) free gates
        :param (B, N) write weighting
        :param (B, N, R) read_weightings

        '''

        usage_write = memory_usage + write_weighting - (memory_usage * write_weighting)
        usage_read = torch.prod( (1 - (read_weightings * free_gates.unsqueeze(1))), dim=2)  # psi # (B, N)
        memory_usage = usage_write * usage_read

        return memory_usage

    def update_free_list(self, memory_usage):
        '''
        Update a (B, N) tensor by sorting memory location by usage in ascending order

        :return free_list (B, N) tensor of sorted indices
        '''

        # sort from smallest to largest
        sorting = torch.sort(memory_usage) # discontinuities in gradient. Not important according to DNC's authors.
        ordered_usage = torch.detach(sorting[0]) # B, N
        free_list = torch.detach(sorting[1].long())

        return free_list, ordered_usage


    def update_allocation_weights(self, sorted_indices, ordered_usage):
        '''
        Update allocation weights (B, N)

        :param (B, R) free gates

        '''

        ones = torch.ones(self.batch_size, device=self.device).view(-1,1)
        usage_ones = torch.cat((ones,ordered_usage[:,:-1]), dim=1)

        prod = torch.cumprod(usage_ones, dim=1)
        #prod = torch.cat((ones, prod), dim=1)

        allocation_weights_ordered = (1 - ordered_usage) * prod

        # reorder allocation weights - inefficient but simpler version
        #for b in self.batch_size:
        #    self.allocation_weights[b][sorted_indices[b]] = allocation_weights_ordered[b]

        # more efficient version than above - avoid a loop over B dimension
        # N.B. this part is crucial for the convergence of the Copy task
        allocation_weights = self.unorder_tensor(allocation_weights_ordered, sorted_indices)

        return allocation_weights


    def unorder_tensor(self, ordered, indices):
        '''
        :param ordered (B, N) tensor with each row in increasing ordered emitted by torch.sort operation
        :param indices (B, N) indices of the ordered tensor emitted by torch.sort operation

        :return allocation_weights (B, N) in the original order of memory usage
        '''

        add = torch.linspace(0, self.batch_size-1, self.batch_size, device=self.device) * indices.size(1)

        indices = indices + add.long().view(-1,1)

        long_ordered = ordered.view(-1)
        long_indices = indices.view(-1)

        allocation_weights_long = torch.zeros_like(long_ordered, device=self.device).float()
        allocation_weights_long[long_indices] = long_ordered

        return allocation_weights_long.view(self.batch_size, -1)


class ContentBasedAddressingRead(nn.Module):
    def __init__(self):
        super(ContentBasedAddressingRead, self).__init__()

    def forward(self, memory, read_strengths, read_vectors):

        '''
        :param (B, N, W) memory
        :param (B, R) read strengths
        :param (B, W, R) read vectors

        :return (B, N, R) tensor that defines a probability distribution
                over memory locations for each read vector.

        '''

        unnormalized_similarity = torch.bmm(memory, read_vectors) # B, N, R
        norms_vectors = torch.norm(read_vectors, 2, dim=1) # B, R
        norms_memory = torch.norm(memory, 2, dim=2) # B, N

        # outer product, each element is the product of the norms of
        # the two corresponding vectors
        normalizer = batch_ger(norms_memory, norms_vectors) # B, N, R

        normalized_similarity = unnormalized_similarity / (normalizer + 1e-8)

        cbr_weightings = F.softmax(read_strengths.unsqueeze(1) * normalized_similarity, dim=1)

        return cbr_weightings


class ContentBasedAddressingWrite(nn.Module):
    def __init__(self):
        super(ContentBasedAddressingWrite, self).__init__()

    def forward(self, memory, write_strength, write_vector):
        '''
        :param (B, N, W) memory
        :param (B) write strength
        :param (B, W) write vector

        :return (B, N) tensor that defines a probability distribution over memory locations.

        '''

        # memory * key^T
        unnormalized_similarity = torch.bmm(memory, write_vector.unsqueeze(2)).squeeze() # B, N
        norm_vector = torch.norm(write_vector,2,dim=1) # B
        norms_memory = torch.norm(memory,2,dim=2) # B, N. Norms of each row of the matrix

        normalized_similarity = unnormalized_similarity / ((norm_vector.view(-1,1) * norms_memory) + 1e-8)

        similarity = write_strength.view(-1,1) * normalized_similarity

        cbw_weightings = F.softmax(similarity, dim=1)

        return cbw_weightings


class Memory(nn.Module):
    def __init__(self, N, W, num_read_heads, batch_size, device):
        super(Memory,self).__init__()

        self.N = N # number of memory slots
        self.W = W # memory slot size
        self.num_read_heads = num_read_heads # R
        self.device = device
        self.batch_size = batch_size

        self.cbw = ContentBasedAddressingWrite()
        self.cbr = ContentBasedAddressingRead()

        self.tml = TemporalMemoryLinkage(self.N, self.batch_size, self.device)
        self.dma = DynamicMemoryAllocation(self.N, self.batch_size, self.device)

        self.reset()

    def reset(self):
        '''
        Reset memory parameters
        '''

        link_matrix, precedence_weighting = self.tml.reset()
        memory_usage = self.dma.reset()

        # memory matrix
        memory = torch.zeros(self.batch_size, self.N,self.W, device=self.device, requires_grad=True)

        # read and write weightings
        read_weightings = torch.zeros(self.batch_size, self.N, self.num_read_heads, device=self.device, requires_grad=True)
        write_weighting = torch.zeros(self.batch_size, self.N, device=self.device, requires_grad=True)

        # read vectors
        read_vectors = torch.zeros(self.batch_size, self.W, self.num_read_heads, device=self.device, requires_grad=True)

        memory_state = [read_vectors, memory, read_weightings, write_weighting, memory_usage, link_matrix, precedence_weighting]

        return memory_state

    def forward(self, erase_vector, free_gates, allocation_gate, write_gate, read_modes,
                read_strengths, read_keys, write_vector, write_key, write_strength, memory_state):

        '''
        Update memory (B, N, W).

        :param (B, W) write vector
        :param (B, W) erase vector

        '''

        read_vectors, memory, read_weightings, write_weighting, memory_usage, link_matrix, precedence_weighting = memory_state

        allocation_weights, memory_usage = self.dma(memory_usage, free_gates, write_weighting, read_weightings)

        cbw_weightings = self.cbw(memory, write_strength, write_key)

        write_weighting = self.update_write_weighting(allocation_gate, write_gate, allocation_weights, cbw_weightings)

        # torch.ger(x,y) = x * y^T
        erase_memory = 1 - batch_ger(write_weighting, erase_vector)
        write_memory = batch_ger(write_weighting, write_vector)
        memory = memory * erase_memory + write_memory

        forward, backward, link_matrix, precedence_weighting = self.tml(link_matrix, precedence_weighting, write_weighting, read_weightings)

        cbr_weightings = self.cbr(memory, read_strengths, read_keys)

        read_weightings = self.update_read_weightings(read_modes, cbr_weightings, forward, backward)

        '''
        Update read vectors (B, W, R).
        One read vector per batch column.
        '''

        read_vectors = torch.bmm(memory.transpose(1,2), read_weightings)

        memory_state = [memory, read_weightings, write_weighting, memory_usage, link_matrix, precedence_weighting]

        return read_vectors, memory_state



    def update_write_weighting(self, allocation_gate, write_gate, allocation_weights, cbw_weightings):

        first = allocation_gate.view(-1,1) * allocation_weights
        second = (1 - allocation_gate).view(-1,1) * cbw_weightings

        write_weighting = write_gate.view(-1,1) * (first + second)

        return write_weighting


    def update_read_weightings(self, read_modes, cbr_weightings, forward_weightings, backward_weightings):
        first = read_modes[:,0,:].unsqueeze(1) * backward_weightings
        second = read_modes[:,1,:].unsqueeze(1) * cbr_weightings
        third = read_modes[:,2,:].unsqueeze(1) * forward_weightings

        read_weightings = first + second + third

        return read_weightings
