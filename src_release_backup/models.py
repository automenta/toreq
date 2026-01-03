"""
Minimal EqProp Models for Reproducibility

This file contains the essential model implementations:
- BackpropMLP: Standard feedforward network (baseline)
- LoopedMLP: Equilibrium Propagation with spectral normalization
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class BackpropMLP(nn.Module):
    """Standard backprop-trained MLP baseline."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class LoopedMLP(nn.Module):
    """
    Equilibrium Propagation MLP with spectral normalization.
    
    Key insight: Spectral norm ensures Lipschitz L < 1, which guarantees
    the free phase converges to a unique fixed point.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, use_spectral_norm=True):
        super().__init__()
        
        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)
        
        # Recurrent layer (this is where equilibrium happens)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.W_out = nn.Linear(hidden_dim, output_dim)
        
        # Apply spectral normalization for stability
        if use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)
        
        self.activation = nn.Tanh()  # Bounded activation for stability
    
    def forward(self, x, steps=20):
        """Forward pass finds equilibrium through iteration."""
        # Initialize hidden state
        h = self.activation(self.W_in(x))
        
        # Iterate to equilibrium (free phase)
        for _ in range(steps):
            h_new = self.activation(self.W_in(x) + self.W_rec(h))
            h = h_new
        
        return self.W_out(h)
    
    def forward_equilibrium(self, x, steps=20):
        """Alias for forward (compatibility)."""
        return self.forward(x, steps)
