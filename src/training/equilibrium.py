import torch

class EquilibriumSolver:
    """
    Manages the equilibrium propagation dynamics.
    Finds the fixed point h* such that h_{t+1} approx h_t.
    """
    def __init__(self, epsilon=1e-4, max_steps=50):
        self.epsilon = epsilon
        self.max_steps = max_steps

    def solve(self, model, x, h_init=None, nudging=False, target_grads=None, beta=0.0):
        """
        Run dynamics until convergence.
        
        Args:
            model: The Looped/Toroidal model.
            x: Input.
            h_init: Initial state (optional).
            nudging: If True, apply nudging.
            target_grads: Gradient of Loss w.r.t h (for nudging).
            beta: Nudging strength.
        
        Returns:
            h_star: Converting state.
            info: Dict with steps taken, residual.
        """
        # We need a unified interface for model.forward_step
        # Assuming model has forward_step(h, x, ...)
        
        batch_size = x.size(0)
        
        if h_init is None:
            h = torch.zeros(batch_size, model.hidden_dim, device=x.device)
        else:
            h = h_init.clone()
            
        params_frozen = True # We don't track gradients through the loop for the fixpoint
        # BUT for EqProp, we might need gradients for the nudging calculation? 
        # Actually EqProp "Free Phase" doesn't need autograd trace. 
        # "Nudged Phase" needs to compute dL/dh to nudge? Or is that given?
        # The README says: h_{t+1} <- (1-alpha)h + alpha*f(h) - beta * dL/dh.
        # dL/dh depends on the OutputHead.
        
        # Check if model requires buffer (Toroidal)
        use_buffer = hasattr(model, 'buffer_size')
        buffer = []

        with torch.no_grad(): # Dynamics are usually standard inference
            for t in range(self.max_steps):
                h_prev = h
                
                if use_buffer:
                    h = model.forward_step(h, x, buffer)
                    buffer.insert(0, h_prev.detach())
                    if len(buffer) > model.buffer_size:
                        buffer.pop()
                else:
                    h = model.forward_step(h, x)
                
                # Apply Nudging if requested
                if nudging and target_grads is not None:
                    # h <- h - beta * dL/dh
                    # For simple classification, L = CrossEntropy(Head(h), y)
                    # We usually pass the gradients explicitly or compute them here.
                    # If we compute here, we need grad enabled for just the head.
                    h.sub_(beta * target_grads)

                # Convergence check
                diff = torch.norm(h - h_prev, dim=1).max()
                if diff < self.epsilon:
                    return h, {"steps": t+1, "converged": True, "buffer": buffer}

        return h, {"steps": self.max_steps, "converged": False, "buffer": buffer}
