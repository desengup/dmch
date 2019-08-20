import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import Linear

def _sequential_allocation(p, weights):
    _, slots, bidders_plus_one = p.shape
    bidders = bidders_plus_one - 1
    
    # total probability of allocating to slot 0
    cumulative_total = p[:,0,:bidders]
    
    # weighted total allocation
    if weights is None:
        alloc = cumulative_total
    else:
        alloc = cumulative_total * weights[0]
    
    for k in range(1,slots):
        # total probability of allocating to slot k
        slot_total = (1-cumulative_total)*p[:,k,:bidders]*(1-p[:,k-1,[bidders for _ in range(bidders)]])

        # weighted total allocation
        if weights is None:
            alloc = alloc + slot_total
        else:
            alloc = alloc + slot_total * weights[k]
        
        cumulative_total = cumulative_total + slot_total
    return alloc
    
class Allocation(Module):
    r"""Determines allocation probability for each of the bidders given an input.
    
    Args:
        in_features: size of each input sample
        bidders: number of bidders, which governs the size of each output sample
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{bidders}`.

    Examples::
        >>> m = Allocation(20, 30)
        >>> input = torch.randn(128, 20)
        >>> allocation = m(input)
        >>> print(allocation.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'bidders']
    
    def __init__(self, in_features, bidders):
        super(Allocation, self).__init__()
        self.in_features = in_features
        self.bidders = bidders
        self.linear = Linear(in_features, bidders+1)
        
    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)[:, 0:self.bidders]

class SequentialAllocation(Module):
    __constants__ = ['in_features', 'bidders', 'slots', 'weights']
    
    def __init__(self, in_features, slots, bidders, weights=None):
        super(SequentialAllocation, self).__init__()
        self.in_features = in_features
        self.slots = slots
        self.bidders = bidders
        self.weights = weights
        self.linear = Linear(in_features, slots * (bidders+1))
        
    def forward(self, x):
        probs = F.softmax(self.linear(x).reshape(-1, self.slots, self.bidders+1), dim=2)
        return _sequential_allocation(probs,weights=self.weights)

class Payment(Module):
    r"""Determines the contingent payment for each of the bidders given an input.
    
    Args:
        in_features: size of each input sample
        bidders: number of bidders, which governs the size of each output sample
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{bidders}`.

    Examples::
        >>> m = Allocation(20, 30)
        >>> input = torch.randn(128, 20)
        >>> payment = m(input)
        >>> print(payment.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features, bidders):
        super(Payment, self).__init__()
        self.in_features = in_features
        self.bidders = bidders
        self.linear = Linear(in_features, bidders)
        
    def forward(self, x):
        return self.linear(x)
    
class Mechanism(Module):
    r"""Determines the allocation and payment of the bidders for a given input.
    
    Args:
        allocation: the network govering allocation
        payment: the network governing payment

    """
    def __init__(self, allocation, payment):
        super(Mechanism, self).__init__()
        self.allocation = allocation
        self.payment = payment
        
    def forward(self, x):
        allocation = self.allocation(x)
        return allocation, allocation*self.payment(x)
    
