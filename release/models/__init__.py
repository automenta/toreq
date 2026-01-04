"""TorEqProp Release Models Package"""

from .looped_mlp import LoopedMLP, BackpropMLP
from .ternary import TernaryEqProp
from .neural_cube import NeuralCube
from .lazy_eqprop import LazyEqProp, LazyStats
from .feedback_alignment import FeedbackAlignmentEqProp, FeedbackAlignmentLayer
from .kernel import EqPropKernel, compare_memory_autograd_vs_kernel

__all__ = [
    'LoopedMLP', 'BackpropMLP',
    'TernaryEqProp',
    'NeuralCube',
    'LazyEqProp', 'LazyStats',
    'FeedbackAlignmentEqProp', 'FeedbackAlignmentLayer',
    'EqPropKernel', 'compare_memory_autograd_vs_kernel',
]
