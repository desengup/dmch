r"""
Deep Mechanism Components

The dmch package contains data structures and algorithms for deep mechanism design.

The package uses pytorch for underlying tensor operations.
"""


import torch

__author__       = 'Patrick R. Jordan'
__email__        = 'patrick.r.jordan@gmail.com'
__version__      = '0.1.0'
__url__          = 'https://github.com/pjordan/dmch/',
__description__  = 'Deep Mechanism Design Components'

from .common import to_inputs, from_inputs
from .mechanism_modules import Allocation, SequentialAllocation, Payment, Mechanism
from .builders import build_allocation_rule, build_payment_rule, build_mechanism, build_spa
from .sequential import SequentialMechanism
from .spa import create_spa_mechanism
from .training import train, evaluate