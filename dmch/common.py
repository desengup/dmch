import torch

def to_inputs(bids, context):
    return torch.cat((bids,context), dim=1)

def from_inputs(inputs, bidders):
    return torch.split(inputs, (bidders,inputs.shape[1]-bidders), dim=1)
    
def utility(allocation, payment, values):
    return allocation*values-payment

def revenue(payment):
    return payment.sum(dim=-1)

def welfare(allocation, values):
    return (allocation * values).sum(dim=-1)
