"""TorEqProp Model Zoo - All Equilibrium Propagation variants."""

# Core models
from .looped_mlp import LoopedMLP
from .bp_mlp import BackpropMLP
from .toroidal_mlp import ToroidalMLP
from .gated_mlp import GatedMLP
from .modern_eqprop import ModernEqProp

# Base class
from .base_eqprop import BaseEqProp

# IDEA implementations
from .spectral_toreqprop import SpectralTorEqProp
from .diff_toreqprop import DiffTorEqProp
from .tp_eqprop import TPEqProp
from .toreq_ode_prop import TorEqODEProp
from .tcep import TCEP
from .mstep import MSTEP
from .tep_ssr import TEPSSR
from .htsep import HTSEP

__all__ = [
    # Core
    "LoopedMLP", "BackpropMLP", "ToroidalMLP", "GatedMLP", "ModernEqProp",
    # Base
    "BaseEqProp",
    # IDEA variants
    "SpectralTorEqProp",  # FFT-based dynamics
    "DiffTorEqProp",      # Diffusion-enhanced
    "TPEqProp",           # Predictive coding
    "TorEqODEProp",       # Continuous ODE
    "TCEP",               # Continuous with recirculation
    "MSTEP",              # Multi-scale pyramid
    "TEPSSR",             # State-space model
    "HTSEP",              # Hyper-toroidal stochastic
]

# Model registry for easy lookup
MODEL_REGISTRY = {
    # Baselines
    'ModernEqProp': ModernEqProp,
    'LoopedMLP': LoopedMLP,
    'ToroidalMLP': ToroidalMLP,
    'GatedMLP': GatedMLP,
    'BackpropMLP': BackpropMLP,
    # IDEA variants
    'SpectralTorEqProp': SpectralTorEqProp,
    'DiffTorEqProp': DiffTorEqProp,
    'TPEqProp': TPEqProp,
    'TorEqODEProp': TorEqODEProp,
    'TCEP': TCEP,
    'MSTEP': MSTEP,
    'TEPSSR': TEPSSR,
    'HTSEP': HTSEP,
}


def get_model(name, **kwargs):
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
