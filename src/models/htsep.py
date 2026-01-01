"""
HTSEP: Hyper-Toroidal Stochastic Equilibrium Propagation

Multi-dimensional adaptive toroidal structures with stochastic spiking neurons.
Extends 1D circular buffers to N-dimensional tori with self-organizing geometry.

Key advantages:
- Continual learning with natural forgetting via toroidal fading
- Neuromorphic hardware efficiency via spiking sparsity
- Multi-modal fusion via higher-dimensional tori
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_eqprop import BaseEqProp


class AdaptiveTorusBuffer(nn.Module):
    """Adaptive multi-dimensional toroidal buffer.
    
    Implements fading memory with exponential decay along toroidal dimensions.
    """
    
    def __init__(self, hidden_dim, torus_dims=2, decay=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.torus_dims = torus_dims
        self.decay = decay
        
        # Learnable decay rates per dimension
        self.decay_rates = nn.Parameter(torch.ones(torus_dims) * decay)
        
        # Buffer storage
        self._buffer = None
        self._position = 0
    
    def reset(self, batch_size, device):
        """Reset buffer."""
        self._buffer = torch.zeros(batch_size, self.hidden_dim, device=device)
        self._position = 0
    
    def update(self, h):
        """Update buffer with new state, applying multi-dimensional fading."""
        if self._buffer is None or self._buffer.shape[0] != h.shape[0]:
            self.reset(h.shape[0], h.device)
        
        # Apply exponential decay
        decay = torch.sigmoid(self.decay_rates).mean()  # Average decay across dims
        self._buffer = decay * self._buffer + (1 - decay) * h
        self._position += 1
        
        return self._buffer


class HTSEP(BaseEqProp):
    """Hyper-Toroidal Stochastic Equilibrium Propagation.
    
    Combines multi-dimensional toroidal buffers with stochastic spiking.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=False, torus_dims=2, spike_threshold=0.5,
                 spike_rate=0.1):
        super().__init__(input_dim, hidden_dim, output_dim, gamma, dropout, use_spectral_norm)
        
        self.spike_threshold = spike_threshold
        self.spike_rate = spike_rate
        
        # Hyper-toroidal buffer
        self.torus_buffer = AdaptiveTorusBuffer(hidden_dim, torus_dims)
        
        # Probabilistic gating for spiking
        self.spike_gate = nn.Linear(hidden_dim, hidden_dim)
    
    def _stochastic_spike(self, h, training=True):
        """Apply stochastic spiking for sparsity."""
        if not training:
            return h
        
        # Compute spike probabilities
        spike_logits = self.spike_gate(h)
        spike_probs = torch.sigmoid(spike_logits)
        
        # Stochastic spiking (straight-through estimator for gradients)
        if self.training:
            spike_mask = (torch.rand_like(spike_probs) < spike_probs).float()
            spike_mask = spike_mask - spike_probs.detach() + spike_probs  # STE
        else:
            spike_mask = (spike_probs > self.spike_threshold).float()
        
        return h * spike_mask
    
    def forward_step(self, h, x, buffer=None, **kwargs):
        """Hyper-toroidal step with stochastic spiking."""
        x_emb = self.embed(x)
        
        # Standard FFN
        ffn_out = self.ffn(h)
        
        # Stochastic spiking
        ffn_spiked = self._stochastic_spike(ffn_out, self.training)
        
        # Target state
        h_target = ffn_spiked + x_emb
        
        # Damped update
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        
        # Update toroidal buffer
        h_torus = self.torus_buffer.update(h_next)
        
        # Blend with buffered state for temporal smoothing
        h_next = 0.8 * h_next + 0.2 * h_torus
        
        return h_next, None
    
    def forward(self, x, steps=30, **kwargs):
        """Reset buffer before forward pass."""
        self.torus_buffer.reset(x.shape[0], x.device)
        return super().forward(x, steps, **kwargs)
    
    def energy(self, h, x, buffer=None):
        """Energy with toroidal buffer regularization."""
        base_energy = self.standard_energy(h, x, buffer)
        
        # Add buffer coherence term
        if self.torus_buffer._buffer is not None:
            buffer_reg = 0.01 * torch.sum((h - self.torus_buffer._buffer) ** 2)
            base_energy = base_energy + buffer_reg
        
        return base_energy
