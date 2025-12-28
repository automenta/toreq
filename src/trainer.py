import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict
from .solver import EquilibriumSolver

class EqPropTrainer:
    """Equilibrium Propagation training loop.
    
    Supports two update mechanisms:
    - 'mse_proxy': MSE loss between model output and nudged equilibrium
    - 'vector_field': Vector field backprop through equilibrium (theoretically cleaner)
    """

    def __init__(self, model: nn.Module, solver: EquilibriumSolver, output_head: nn.Module, 
                 beta: float = 0.1, lr: float = 1e-3, update_mode: str = 'mse_proxy'):
        self.model = model
        self.solver = solver
        self.output_head = output_head
        self.beta = beta
        self.update_mode = update_mode
        
        if update_mode not in ['mse_proxy', 'vector_field']:
            raise ValueError(f"update_mode must be 'mse_proxy' or 'vector_field', got {update_mode}")
        
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(output_head.parameters()),
            lr=lr
        )

    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Single training step using Equilibrium Propagation.
        
        Nudge Sign Convention:
        - We compute grads = ∇L (gradient of loss)
        - Nudge is applied as: h_new - β * grads
        - This moves h in the direction that decreases loss
        - Equivalent to: h_new + β * ∇(-L)
        
        Args:
            x: Input [seq, batch, d_model]
            y: Target labels [batch]
            
        Returns:
            Dictionary with loss, accuracy, and iteration counts
        """
        # Free phase: Find equilibrium without nudging
        h0 = torch.zeros_like(x)
        with torch.no_grad():
            h_free, iters_free = self.solver.solve(self.model, h0, x)

        # Nudged phase: Find equilibrium with loss-based nudging
        def nudged_dynamics(h, x):
            h = h.detach().requires_grad_(True)
            h_new = self.model(h, x)
            y_pred = self.output_head(h_new.mean(dim=0))
            loss = F.cross_entropy(y_pred, y)
            
            # Nudge in direction that decreases loss
            grads = torch.autograd.grad(loss, h_new, create_graph=True, retain_graph=True)[0]
            return h_new - self.beta * grads

        h_nudged, iters_nudged = self.solver.solve(nudged_dynamics, h_free.detach(), x)

        # Update model parameters
        self.optimizer.zero_grad()

        if self.update_mode == 'mse_proxy':
            # MSE Proxy Loss approach
            # Minimize distance between model output and nudged equilibrium
            h_free_detached = h_free.detach()
            h_out = self.model(h_free_detached, x)
            
            # Scale by 1/beta for correct gradient magnitude
            # Theory: delta ~ 1/beta * (h_free - h_nudged) approximates adjoint state
            loss_proxy = (1.0 / self.beta) * F.mse_loss(h_out, h_nudged.detach())
            
            # Output head update: standard gradient at free equilibrium
            y_pred_free = self.output_head(h_free_detached.mean(dim=0))
            loss_head = F.cross_entropy(y_pred_free, y)
            
            total_loss = loss_proxy + loss_head
            total_loss.backward()
            
        else:  # update_mode == 'vector_field'
            # Vector Field Backprop approach
            # Gradient ≈ 1/beta * (h_nudged - h_free) * df/dθ
            # We backpropagate v = (h_nudged - h_free) / beta through f(h_free)
            
            delta = h_nudged - h_free.detach()
            v = delta / self.beta
            
            # Re-compute output at free fixed point with gradients enabled
            h_at_free = h_free.detach()
            with torch.enable_grad():
                out_free = self.model(h_at_free, x)
            
            # Backward the vector v through model parameters
            out_free.backward(gradient=v)
            
            # Output head update: standard gradient at free equilibrium
            with torch.enable_grad():
                y_pred_free = self.output_head(h_free.mean(dim=0))
                loss_head = F.cross_entropy(y_pred_free, y)
                loss_head.backward()
            
            total_loss = loss_head  # For logging

        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            y_pred_free = self.output_head(h_free.mean(dim=0))
            acc = (y_pred_free.argmax(-1) == y).float().mean()

        return {
            "loss": total_loss.item(),
            "accuracy": acc.item(),
            "iters_free": iters_free,
            "iters_nudged": iters_nudged
        }
