import torch
import torch.nn as nn
from torch.nn import Module
from .common import to_inputs
from .common import from_inputs

def _dot(alist,blist):
    return sum(a*b for a,b in zip(alist,blist))

def cascade_outcomes(allocators, pricers, inputs, bidders):
    bids, context = from_inputs(inputs, bidders)
    allocations, prices = [], []
    cumulative_allocation = None
    prev_allocatinon_prob = None
    
    for allocator, pricer in zip(allocators, pricers):
        slot_inputs = to_inputs(bids, context)
        allocation = allocator(slot_inputs)
        price = pricer(slot_inputs)
        if cumulative_allocation is None:
            allocations.append(allocation)
            prices.append(price)
            cumulative_allocation = allocation
            prev_allocation_prob = torch.cat([allocation.sum(dim=1,keepdim=True) for _ in range(bidders)], dim=1)
        else:
            unconditional_allocation = allocation * (1-cumulative_allocation)*prev_allocation_prob
            allocations.append(unconditional_allocation)
            prices.append(price)
            cumulative_allocation = cumulative_allocation + unconditional_allocation
            prev_allocation_prob = torch.cat([unconditional_allocation.sum(dim=1,keepdim=True) for _ in range(bidders)], dim=1)
        bids = (1-cumulative_allocation) * bids
    return allocations, prices
    
class SequentialMechanism(Module):
    r"""Determines the allocation and payment of the bidders for a given input that allows sequential allocation.
    
    Args:
        mechanisms: the networks govering allocation
        
    """
    def __init__(self, allocators, pricers, bidders, weights=None, virtual_fcn=None):
        super(SequentialMechanism, self).__init__()
        self.allocators = nn.ModuleList(allocators)
        self.pricers = nn.ModuleList(pricers)
        self.weights = weights
        self.bidders = bidders
        self.virtual_fcn = virtual_fcn
        
    def _compute_virtual_bids(self,bids,context):
        return torch.cat(
            [self.virtual_fcn(to_inputs(bids[:,i:(i+1)],context)) for i in range(self.bidders)],
            dim=1)
    
    def _compute_prices(self,vprices,bids,context):
        return torch.cat(
            [self.virtual_fcn(to_inputs(vprices[:,i:(i+1)],context),bids=bids[:,i:(i+1)],invert=True) for i in range(self.bidders)],
            dim=1)
    
    def forward(self, x):
        if self.virtual_fcn is None:
            allocations, prices = cascade_outcomes(self.allocators, self.pricers, x, self.bidders)
        else:
            bids, context = from_inputs(x, self.bidders)
            vbids = self._compute_virtual_bids(bids, context)
            vx = to_inputs(vbids,context)
            allocations, vprices = cascade_outcomes(self.allocators, self.pricers, vx, self.bidders)
            prices = [self._compute_prices(vprices[i], bids, context) for i in range(len(vprices))]
            
        payments = [a*p for a,p in zip(allocations, prices)]
        if self.weights:
            return _dot(self.weights, allocations), _dot(self.weights, payments)
        else:
            return sum(allocations), sum(payments)