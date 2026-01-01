"""
TorEqODEProp: Toroidal Equilibrium ODE Propagation

Models relaxation as a continuous-time Neural ODE on a toroidal manifold.
States evolve via continuous dynamics until equilibrium (dh/dt ≈ 0).

Key advantages:
- Infinite effective depth without parameter explosion
- Better for cyclic/periodic data via toroidal constraints
- Continuous flow + finite-nudge reduces oscillations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_eqprop import BaseEqProp


class TorEqODEProp(BaseEqProp):
    """Toroidal Equilibrium ODE Propagation.
    
    Uses simple Euler integration to approximate ODE dynamics:
    dh/dt = -∇_h E(h; x, θ)
    
    With toroidal projection: h mod 2π in toroidal dimensions.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=False, dt=0.1, damping=0.5, toroidal_dims=0):
        super().__init__(input_dim, hidden_dim, output_dim, gamma, dropout, use_spectral_norm)
        
        self.dt = dt  # Integration timestep
        self.damping = damping  # Damping coefficient λ
        self.toroidal_dims = toroidal_dims  # Number of dimensions to make toroidal
    
    def _toroidal_project(self, h):
        """Project specified dimensions onto torus (periodic boundaries)."""
        if self.toroidal_dims > 0:
            # Apply modular arithmetic to first toroidal_dims dimensions
            h_tor = h.clone()
            h_tor[..., :self.toroidal_dims] = torch.remainder(
                h[..., :self.toroidal_dims], 2 * torch.pi
            )
            return h_tor
        return h
    
    def forward_step(self, h, x, buffer=None, **kwargs):
        """Euler integration step for ODE dynamics."""
        x_emb = self.embed(x)
        
        # Compute f_θ(h, x) - the network function
        ffn_out = self.ffn(h)
        
        # ODE: dh/dt = f_θ(h, x) - λh (damped dynamics)
        dh_dt = ffn_out + x_emb - self.damping * h
        
        # Euler step: h(t+dt) = h(t) + dt * dh/dt
        h_next = h + self.dt * dh_dt
        
        # Apply toroidal projection
        h_next = self._toroidal_project(h_next)
        
        return h_next, None
    
    def energy(self, h, x, buffer=None):
        """Standard energy for ODE dynamics."""
        return self.standard_energy(h, x, buffer)
