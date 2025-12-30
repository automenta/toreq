import torch
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class DynamicsProfile:
    avg_lyapunov: float
    convergence_rate: float
    is_chaotic: bool
    energy_history: List[float]

class DynamicsAnalyzer:
    """Analyzes the dynamical properties of Equilibrium Propagation.
    
    EqProp is a physical system that settles to equilibrium.
    This analyzer measures:
    1. Lyapunov Exponents: Rate of divergence/convergence.
    2. Phase Transitions: Detecting boundaries between order and chaos.
    """
    
    def __init__(self, model):
        self.model = model
    
    def analyze_trajectory(self, x: torch.Tensor, steps: int = 50) -> DynamicsProfile:
        """Track the trajectory of the hidden states during inference."""
        
        # Assuming model exposes state access
        # This will be model-specific, so we might need an interface
        # For now, we assume model.step(x, state) exists
        
        # Lyapunov estimation via perturbation
        perturbation_scale = 1e-6
        
        # 1. Run clean trajectory
        states_clean = self._collect_trajectory(x, steps)
        
        # 2. Run perturbed trajectory
        # Add tiny noise to INITIAL hidden state (or input)
        x_perturbed = x + torch.randn_like(x) * perturbation_scale
        states_perturbed = self._collect_trajectory(x_perturbed, steps)
        
        # 3. Calculate divergence
        divergences = []
        for s1, s2 in zip(states_clean, states_perturbed):
            dist = torch.norm(s1 - s2).item()
            divergences.append(dist)
        
        # 4. Fit exponent: dist(t) ~ dist(0) * e^(lambda * t)
        # log(dist(t)) ~ log(dist(0)) + lambda * t
        # lambda is the slope of log(dist) vs t
        
        # Avoid log(0)
        log_divs = np.log(np.array(divergences) + 1e-12)
        ts = np.arange(len(log_divs))
        
        # Simple linear regression for slope
        slope, intercept = np.polyfit(ts, log_divs, 1)
        
        # Energy tracking (if model supports it)
        energies = []
        if hasattr(self.model, "energy"):
            for s in states_clean:
                energies.append(self.model.energy(x, s).item())
        
        return DynamicsProfile(
            avg_lyapunov=slope,
            convergence_rate=-slope if slope < 0 else 0, # Positive convergence rate
            is_chaotic=(slope > 0),
            energy_history=energies
        )
    
    def _collect_trajectory(self, x: torch.Tensor, steps: int) -> List[torch.Tensor]:
        """Collect hidden states over T steps."""
        # This implementation depends heavily on the model structure.
        # We need a generic way to step the model.
        # Mocking for now - expected to be updated when integrated with specific model class
        return [torch.zeros(1) for _ in range(steps)]

