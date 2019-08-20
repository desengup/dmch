import torch
import torch.nn as nn
from . import Mechanism
from . import from_inputs

class _SpaAllocation(nn.Module):
    def __init__(self, bidders, kappa=1e4):
        super(_SpaAllocation, self).__init__()
        self.kappa = kappa
        self.bidders = bidders
        
    def forward(self, x):
        bids, _ = from_inputs(x, self.bidders)
        device = bids.get_device()
        bids_plus_dummy = torch.cat([bids, torch.zeros(bids.shape[0],1).to(device)], dim=1)
        return torch.softmax(self.kappa*bids_plus_dummy, dim=1)[:,:-1]

class _SpaPayment(nn.Module):
    def __init__(self, bidders):
        super(_SpaPayment, self).__init__()
        self.bidders = bidders
    def forward(self, x):
        bids, _ = from_inputs(x, self.bidders)
        device = x.get_device()
        return torch.stack([bids.index_fill(1,torch.tensor([col]).to(device),0).max(dim=1)[0] for col in range(bids.shape[1])], dim=1)
    
def create_spa_allocator(bidders,kappa=1e4):
    return _SpaAllocation(bidders,kappa=kappa)

def create_spa_pricer(bidders):
    return _SpaPayment(bidders,)

def create_spa_mechanism(bidders,kappa=1e4):
    return Mechanism(
        create_spa_allocator(bidders,kappa=1e4),
        create_spa_pricer(bidders))