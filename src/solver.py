import torch
from torch import Tensor
from typing import Callable, Tuple

class EquilibriumSolver:
    """Fixed-point solver with convergence monitoring."""

    def __init__(self, max_iters: int = 50, tol: float = 1e-5, damping: float = 0.9):
        self.max_iters = max_iters
        self.tol = tol
        self.damping = damping

    def solve(self, f: Callable[[Tensor, Tensor], Tensor], h0: Tensor, x: Tensor) -> Tuple[Tensor, int]:
        h = h0
        for t in range(self.max_iters):
            # h_new = (1 - self.damping) * h + self.damping * f(h, x)
            fx = f(h, x)
            h_new = (1 - self.damping) * h + self.damping * fx

            residual = (h_new - h).norm()
            if residual < self.tol:
                return h_new, t + 1
            h = h_new
        return h, self.max_iters  # Did not converge
