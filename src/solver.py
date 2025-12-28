"""Fixed-point equilibrium solver for equilibrium propagation."""

import torch
from torch import Tensor
from typing import Callable, Tuple


class EquilibriumSolver:
    """Fixed-point solver with damped iteration and convergence monitoring.
    
    Solves for h* such that h* = f(h*, x) using damped fixed-point iteration:
        h_{t+1} = (1 - α) h_t + α f(h_t, x)
    
    where α ∈ (0, 1] is the damping factor.
    """

    def __init__(self, max_iters: int = 50, tol: float = 1e-5, damping: float = 0.9):
        """Initialize solver.
        
        Args:
            max_iters: Maximum number of iterations
            tol: Convergence tolerance (L2 norm of residual)
            damping: Damping factor α ∈ (0, 1]. Higher = less damping.
        """
        self.max_iters = max_iters
        self.tol = tol
        self.damping = damping

    def solve(self, f: Callable[[Tensor, Tensor], Tensor], 
              h0: Tensor, x: Tensor) -> Tuple[Tensor, int]:
        """Solve for equilibrium h* = f(h*, x).
        
        Args:
            f: Dynamics function f(h, x) -> h_new
            h0: Initial state
            x: Input (constant during iteration)
            
        Returns:
            Tuple of (equilibrium state, number of iterations)
        """
        h = h0
        for t in range(self.max_iters):
            fx = f(h, x)
            h_new = (1 - self.damping) * h + self.damping * fx

            residual = (h_new - h).norm()
            if residual < self.tol:
                return h_new, t + 1
            h = h_new
        
        # Did not converge within max_iters
        return h, self.max_iters
