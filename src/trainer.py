import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List
from .solver import EquilibriumSolver

class EqPropTrainer:
    """Equilibrium Propagation training loop."""

    def __init__(self, model: nn.Module, solver: EquilibriumSolver, output_head: nn.Module, beta: float = 0.1, lr: float = 1e-3):
        self.model = model
        self.solver = solver
        self.output_head = output_head
        self.beta = beta
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(output_head.parameters()),
            lr=lr
        )

    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        # x: [seq, batch, d_model]
        # y: [batch]

        # 1. Free Phase
        # We treat model params as constant for finding fixed point
        with torch.no_grad():
            h0 = torch.zeros_like(x)
            h_free, iters_free = self.solver.solve(self.model, h0, x)

        # 2. Nudged Phase
        h_free_detached = h_free.detach()
        h_free_detached.requires_grad_(True)

        def nudged_dynamics(h, x):
            # Calculate dL/dh to nudge
            with torch.enable_grad():
                h_in = h.detach().requires_grad_(True)
                h_out = self.model(h_in, x)
                # Output head on mean pooling
                y_pred = self.output_head(h_out.mean(dim=0))
                loss = F.cross_entropy(y_pred, y)

                # nudge = - beta * dL/dh
                grads = torch.autograd.grad(loss, h_in)[0]

            return h_out - self.beta * grads

        with torch.no_grad():
            h_nudged, iters_nudged = self.solver.solve(nudged_dynamics, h_free_detached, x)

        # 3. Update (Vector Field / Contrastive)
        # Gradient \approx 1/beta * (h_nudged - h_free) * df/dtheta
        # We backpropagate v = 1/beta * (h_nudged - h_free) through f(h_free)
        # Note: In test_gradient_equiv we used v = - delta / beta and got negative correlation (-0.98).
        # This implies true gradient matches + delta / beta.
        # So v = (h_nudged - h_free) / beta.

        delta = h_nudged - h_free_detached
        v = delta / self.beta

        self.optimizer.zero_grad()

        # Re-compute output at free fixed point to connect graph
        h_at_free = h_free_detached # No grad needed for h
        # Enable grad for model parameters
        with torch.enable_grad():
            out_free = self.model(h_at_free, x)

        # Backward the vector v
        out_free.backward(gradient=v)

        # We also need gradient for Output Head!
        # The above only updates 'model' (f).
        # We need dL/d(output_head_params).
        # Standard BP on loss at equilibrium?
        # Scellier 2017: Theta_readout updates with dL/dTheta_readout evaluated at h_free?
        # Yes, usually "readout" layer is trained with standard gradient at free phase.

        with torch.enable_grad():
            y_pred_free = self.output_head(h_free.mean(dim=0))
            loss_main = F.cross_entropy(y_pred_free, y)
            loss_main.backward() # Accumulates gradients into output_head (and model via h_free?)
            # Wait, h_free is detached. So this only updates output_head.
            # Perfect.

        self.optimizer.step()

        with torch.no_grad():
            acc = (y_pred_free.argmax(-1) == y).float().mean()

        return {
            "loss": loss_main.item(),
            "accuracy": acc.item(),
            "iters_free": iters_free,
            "iters_nudged": iters_nudged
        }
