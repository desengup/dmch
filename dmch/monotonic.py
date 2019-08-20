import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from .common import from_inputs


class _ConstantWeightsAndBiases(nn.Module):
    def __init__(self, out_features):
        super(_ConstantWeightsAndBiases, self).__init__()
        self.weights = Parameter(torch.Tensor(out_features, 1))
        self.biases = Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.biases, a=math.sqrt(5))
        
    def forward(self, context):
        return self.weights, self.biases
    
class _VariableWeightsAndBiases(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(_VariableWeightsAndBiases, self).__init__()
        self.linear = nn.Linear(in_features, hidden_features)
        self.weights = nn.Linear(hidden_features, out_features)
        self.biases = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return self.weights(x), self.biases(x)
    
class Monotonic(nn.Module):
    def __init__(self, hidden_features, linear_functions, groups, context_features=0):
        super(Monotonic, self).__init__()
        self.context_features = context_features
        self.linear_functions = linear_functions
        self.groups = groups
        
        if context_features > 0:
            self.weights_and_biases = _VariableWeightsAndBiases(context_features, hidden_features, linear_functions * groups)
        else:
            self.weights_and_biases = _ConstantWeightsAndBiases(linear_functions * groups)
        
        
    def apply_forward(self, bids, context):
        w, b = self.weights_and_biases(context)
        intermediate = torch.exp(w) * bids + b
        return intermediate.reshape(-1, self.groups, self.linear_functions).max(dim=2)[0].min(dim=1, keepdim=True)[0]
        
    def apply_inverse(self, bids, vbids, context):
        w, b = self.weights_and_biases(context)
        intermediate = torch.exp(-w) * (vbids - b)
        return intermediate.reshape(-1, self.groups, self.linear_functions).min(dim=2)[0].max(dim=1, keepdim=True)[0]
            
    def forward(self, inputs, bids=None, invert=False):
        x, context = from_inputs(inputs,1)
        if invert:
            return self.apply_inverse(bids, x, context)
        else:
            return self.apply_forward(x, context)
        
def create_monotonic(context_features=0, hidden_features=1, linear_functions=1, groups=1):
    return Monotonic(hidden_features, linear_functions, groups, context_features=context_features)
