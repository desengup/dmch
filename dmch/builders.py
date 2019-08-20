import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import Linear

from .mechanism_modules import Allocation
from .mechanism_modules import SequentialAllocation
from .mechanism_modules import Payment
from .mechanism_modules import Mechanism

from .spa import create_spa_allocator, create_spa_pricer
from .monotonic import create_monotonic
from .sequential import SequentialMechanism

class _LeftAddition(Module):
    def __init__(self, left_transformation):
        super(_LeftAddition, self).__init__()
        self.left_transformation = left_transformation
        
    def forward(self, x):
        return self.left_transformation(x)+x
    
class _BaseBidResponseBuilder(object):
    def __init__(self, bidders, context_features):
        self.bidders = bidders
        self.context_features = context_features
        self.current_features = bidders + context_features
        self.layers = []
    
    def add_identity(self):
        self.layers.append(nn.Identity())
    
    def add_batch_normalization(self):
        self.layers.append(nn.BatchNorm1d(self.current_features))
    
    def add_batch_normalization(self):
        self.layers.append(nn.BatchNorm1d(self.current_features))
        
    def add_sigmoid_activation(self):
        self.layers.append(nn.Sigmoid())
        
    def add_leaky_relu_activation(self):
        self.layers.append(nn.LeakyReLU())
    
    def add_activation(self, act_module):
        self.layers.append(act_module)
        
    def add_linear_layer(self, out_features):
        self.layers.append(nn.Linear(self.current_features, out_features))
        self.current_features = out_features
        
    def add_residual_layer(self, act_layer=nn.LeakyReLU):
        self.layers.append(
            _LeftAddition(
                nn.Sequential(
                    nn.BatchNorm1d(self.current_features),
                    act_layer(),
                    nn.Linear(self.current_features, self.current_features),
                    nn.BatchNorm1d(self.current_features),
                    act_layer(),
                    nn.Linear(self.current_features, self.current_features)
                )
            )
        )
        
    def build(self):
        pass
    
class _AllocationRuleBuilder(_BaseBidResponseBuilder):    
    def build(self):
        layers = self.layers + [Allocation(self.current_features, self.bidders)]
        return nn.Sequential(*layers)
    
    def build_sequential(self,slots,weights=None):
        layers = self.layers + [SequentialAllocation(self.current_features, slots, self.bidders, weights=weights)]
        return nn.Sequential(*layers)
    
class _PaymentRuleBuilder(_BaseBidResponseBuilder):    
    def build(self):
        layers = self.layers + [Payment(self.current_features, self.bidders)]
        return nn.Sequential(*layers)
        
class _MechanismBuilder(object):
    def __init__(self, bidders, context_features=0):
        self.allocation_builder = build_allocation_rule(bidders, context_features=context_features)
        self.payment_builder = build_payment_rule(bidders, context_features=context_features)
    
    def build(self):
        return Mechanism(
            self.allocation_builder.build(),
            self.payment_builder.build())
    
    def build_sequential(self,slots,weights=None):
        return Mechanism(
            self.allocation_builder.build_sequential(slots,weights=weights),
            self.payment_builder.build())
    
class _SpaBuilder(object):
    def __init__(self, bidders, context_features=0):
        self.bidders = bidders
        self.context_features = context_features
        self.virtual_fcn = None
        
    def set_virtual_function(self, hidden_features=1, linear_functions=1, groups=1):
        self.virtual_fcn = create_monotonic(
            context_features=self.context_features,
            hidden_features=hidden_features,
            linear_functions=linear_functions,
            groups=groups)

    def build(self):
        return create_spa_mechanism(self.bidders, context_features=self.context_features)
    
    def build_sequential(self, slots, weights=None):
        return SequentialMechanism(
            [create_spa_allocator(self.bidders) for _ in range(slots)],
            [create_spa_pricer(self.bidders) for _ in range(slots)],
            self.bidders,
            weights=weights,
            virtual_fcn=self.virtual_fcn
        )
    
def build_allocation_rule(bidders, context_features=0):
    return _AllocationRuleBuilder(bidders, context_features=context_features)

def build_payment_rule(bidders, context_features=0):
    return _PaymentRuleBuilder(bidders, context_features=context_features)

def build_mechanism(bidders, context_features=0):
    return _MechanismBuilder(bidders, context_features=context_features)

def build_spa(bidders, context_features=0):
    return _SpaBuilder(bidders, context_features=context_features)