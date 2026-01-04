"""
Homeostatic EqProp: Self-Tuning Dynamic Lipschitz Scaling (Track 8)

Implements "Autonomic Homeostasis" - a network that monitors its stability
and automatically adjusts weight scales to maintain L < 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from .utils import estimate_lipschitz

@dataclass
class HomeostasisMetrics:
    avg_velocity: float
    lipschitz_estimate: float
    brake_applied: float
    boost_applied: float
    layers_braked: int
    layers_boosted: int

class HomeostaticEqProp(nn.Module):
    """
    EqProp with Dynamic Lipschitz Scaling for autonomous stability.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 5, alpha: float = 0.5,
                 target_lipschitz: float = 0.95,
                 velocity_threshold_high: float = 0.1,
                 velocity_threshold_low: float = 0.01,
                 adaptation_rate: float = 0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha
        
        # Regulation parameters
        self.target_lipschitz = target_lipschitz
        self.velocity_threshold_high = velocity_threshold_high
        self.velocity_threshold_low = velocity_threshold_low
        self.adaptation_rate = adaptation_rate
        
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Learnable scaling factors (adjusted by homeostasis)
        self.register_buffer('layer_scales', torch.ones(num_layers))
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
        # Conservative init
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(0.7)
                
        self.last_velocities: Dict[int, float] = {}
        self.homeostasis_history: List[HomeostasisMetrics] = []

    def _estimate_layer_lipschitz(self, layer_idx: int) -> float:
        """Estimate effective Lipschitz constant of a layer."""
        # Temporarily scale weight
        original_weight = self.layers[layer_idx].weight
        scaled_weight = original_weight * self.layer_scales[layer_idx]
        
        # Create a temporary container to use shared utility
        # Wrapper to match utils interface if needed, but simple linear suffices
        with torch.no_grad():
            W = scaled_weight
            u = torch.randn(W.shape[1], device=W.device)
            u = F.normalize(u, dim=0)
            for _ in range(3):
                v = F.normalize(W @ u, dim=0)
                u = F.normalize(W.T @ v, dim=0)
            sigma = torch.norm(W @ u).item()
        return sigma

    def forward_step(self, h_states: Dict[int, torch.Tensor], x: torch.Tensor,
                     track_velocity: bool = False) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
        """Single equilibrium step."""
        new_states = {}
        velocities = {}
        x_emb = self.W_in(x)
        
        for i, layer in enumerate(self.layers):
            pre = x_emb if i == 0 else h_states.get(i-1, torch.zeros_like(x_emb))
            h_curr = h_states.get(i, torch.zeros_like(pre))
            
            # Apply scaling
            scale = self.layer_scales[i]
            h_target = torch.tanh(F.linear(pre, layer.weight * scale, layer.bias))
            
            h_new = (1 - self.alpha) * h_curr + self.alpha * h_target
            new_states[i] = h_new
            
            if track_velocity:
                velocity = torch.mean(torch.abs(h_new - h_curr)).item()
                velocities[i] = velocity
                
        return new_states, velocities

    def apply_homeostasis(self, velocities: Dict[int, float]) -> HomeostasisMetrics:
        """Apply homeostatic regulation based on velocity."""
        brake_total = 0.0
        boost_total = 0.0
        layers_braked = 0
        layers_boosted = 0
        
        for i, velocity in velocities.items():
            if velocity > self.velocity_threshold_high:
                # Brake
                factor = 1.0 - self.adaptation_rate
                self.layer_scales[i] *= factor
                brake_total += (1.0 - factor)
                layers_braked += 1
            elif velocity < self.velocity_threshold_low:
                # Boost, but respect L limit
                current_L = self._estimate_layer_lipschitz(i)
                if current_L < self.target_lipschitz:
                    factor = 1.0 + self.adaptation_rate
                    self.layer_scales[i] *= factor
                    boost_total += (factor - 1.0)
                    layers_boosted += 1
                    
        self.layer_scales.clamp_(0.1, 2.0)
        
        avg_v = sum(velocities.values()) / len(velocities) if velocities else 0.0
        avg_L = sum(self._estimate_layer_lipschitz(i) for i in range(self.num_layers)) / self.num_layers
        
        metrics = HomeostasisMetrics(avg_v, avg_L, brake_total, boost_total, layers_braked, layers_boosted)
        self.homeostasis_history.append(metrics)
        self.last_velocities = velocities
        return metrics

    def forward(self, x: torch.Tensor, steps: int = 30, 
                apply_homeostasis: bool = True) -> torch.Tensor:
        """Forward pass with auto-regulation."""
        batch_size = x.size(0)
        h_states = {i: torch.zeros(batch_size, self.hidden_dim, device=x.device) 
                    for i in range(self.num_layers)}
        
        all_velocities = []
        for step in range(steps):
            track = step >= steps // 2
            h_states, velocities = self.forward_step(h_states, x, track_velocity=track)
            if track:
                all_velocities.append(velocities)
                
        if apply_homeostasis and all_velocities:
            avg_velocities = {}
            for i in range(self.num_layers):
                avg_velocities[i] = sum(v.get(i, 0) for v in all_velocities) / len(all_velocities)
            self.apply_homeostasis(avg_velocities)
            
        return self.head(h_states[self.num_layers-1])

    def get_stability_report(self) -> str:
        """Generate stability status report."""
        lipschitz = [self._estimate_layer_lipschitz(i) for i in range(self.num_layers)]
        max_L = max(lipschitz) if lipschitz else 0.0
        status = "✓ STABLE" if max_L < 1.0 else "⚠ UNSTABLE"
        
        lines = [
            f"Max Lipschitz: {max_L:.4f} {status}",
            f"Layer Scales: {[f'{s:.3f}' for s in self.layer_scales.tolist()]}"
        ]
        if self.homeostasis_history:
            last = self.homeostasis_history[-1]
            lines.append(f"Last Action: {last.layers_braked} braked, {last.layers_boosted} boosted")
            
        return "\n".join(lines)
