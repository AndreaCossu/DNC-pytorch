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
    

def accuracy(out, target, masks):
    with torch.no_grad():
        
        accuracy_unmasked = (out[:,:,:-1].round() == target[:,:,:-1]).float().sum(dim=2) / float(out.size(2) - 1) # vector_len
    
        accuracy = torch.mv(accuracy_unmasked, masks) / torch.sum(masks).float()
    
        acc_perc = torch.mean(accuracy).item()*100
            
        return acc_perc
