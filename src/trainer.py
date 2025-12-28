import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict
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
        # x: [seq, batch, d_model] assuming model expects that
        # y: [batch] labels

        # Free phase
        h0 = torch.zeros_like(x)
        # We need h_free detached for nudged phase initialization, but attached for loss update?
        # No, we recompute one step of free dynamics for the loss update.
        with torch.no_grad():
             h_free, iters_free = self.solver.solve(self.model, h0, x)

        # Nudged phase
        # We start from h_free (detached)
        h_start = h_free.detach()
        h_start.requires_grad_(True)

        def nudged_dynamics(h, x):
            # We need to compute gradients of L w.r.t h
            # So h must require grad.
            h = h.detach().requires_grad_(True)

            h_new = self.model(h, x)
            # We assume h_new is [seq, batch, d_model]
            # pooling over sequence
            y_pred = self.output_head(h_new.mean(dim=0))

            loss = F.cross_entropy(y_pred, y)
            grads = torch.autograd.grad(-loss, h_new, create_graph=True, retain_graph=True)[0]

            return h_new + self.beta * grads

        h_nudged, iters_nudged = self.solver.solve(nudged_dynamics, h_free.detach(), x)

        # Update Rule: Target Propagation Style
        # Minimize distance between model output (one step from h_free) and h_nudged (target).
        # We need to attach h_free to the graph for the model parameters.
        # h_out = model(h_free.detach(), x)
        # loss = MSE(h_out, h_nudged.detach())

        h_free_detached = h_free.detach()
        h_out = self.model(h_free_detached, x)

        # Scale loss by 1/beta as per EqProp theory (delta ~ 1/beta * (h_free - h_nudged))
        # MSE gradient is 2 * (h_out - target).
        # We want gradient ~ (h_free - h_nudged) / beta.
        # So we use (1/beta) * MSE ?
        # Actually (1/(2*beta)) * MSE gives gradient (1/beta)*(h_out - target).
        # Since h_out approx h_free.

        loss_proxy = (1.0 / self.beta) * F.mse_loss(h_out, h_nudged.detach())

        # Output head update
        # Standard gradient at free phase equilibrium
        y_pred_free = self.output_head(h_free_detached.mean(dim=0))
        loss_head = F.cross_entropy(y_pred_free, y)

        total_loss = loss_proxy + loss_head

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            acc = (y_pred_free.argmax(-1) == y).float().mean()

        return {
            "loss": total_loss.item(),
            "accuracy": acc.item(),
            "iters_free": iters_free,
            "iters_nudged": iters_nudged
        }
