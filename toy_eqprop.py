import torch
import torch.nn as nn
import torch.nn.functional as F
from src.solver import EquilibriumSolver

class ToyEnergyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)
        # Enforce symmetry manually
        self.W.weight.data = (self.W.weight.data + self.W.weight.data.T) / 2

    def forward(self, h, x):
        # Symmetric dynamics: h = tanh(W h + x)
        # Use functional linear with symmetric weight
        w_sym = (self.W.weight + self.W.weight.t()) / 2
        return torch.tanh(F.linear(h, w_sym, self.W.bias) + x)

def test_toy_eqprop():
    print("Testing Toy EqProp on Symmetric RNN...")
    dim = 10
    model = ToyEnergyModel(dim)

    x = torch.randn(1, dim)
    y = torch.randint(0, 2, (1,))
    target = torch.randn(1, dim) # Regression for simplicity

    solver = EquilibriumSolver(max_iters=200, tol=1e-6, damping=0.5)
    beta = 0.001

    # 1. Free Phase
    h0 = torch.zeros_like(x)
    with torch.no_grad():
        h_free, _ = solver.solve(model, h0, x)

    # 2. BP Gradient
    # Unroll
    h = torch.zeros_like(x, requires_grad=True)
    for _ in range(200):
        h = (1-0.5)*h + 0.5*model(h, x)

    loss_bp = 0.5 * ((h - target)**2).sum()
    loss_bp.backward()

    grad_bp = model.W.weight.grad.clone()
    print(f"BP Grad Norm: {grad_bp.norm().item()}")

    # 3. EqProp
    model.zero_grad()
    h_free_detached = h_free.detach().clone().requires_grad_(True)

    def nudged_dynamics(h, x):
        # Nudge towards target
        # Loss = 0.5 * (h - target)^2
        # dL/dh = h - target
        with torch.enable_grad():
            h_in = h.detach().requires_grad_(True)
            loss = 0.5 * ((h_in - target)**2).sum()
            g = torch.autograd.grad(loss, h_in)[0]

        # Nudge: - beta * dL/dh
        # Note: In standard EqProp for Energy E, h_dot = -dE/dh - beta dL/dh
        # Dynamics: h_new = h - dE/dh = f(h) (approx).
        # Actually h_dot = -h + f(h).
        # So h_dot = -h + f(h) - beta dL/dh.
        # h_new = (1-a)h + a(f(h) - beta dL/dh).

        return model(h_in, x) - beta * g

    with torch.no_grad():
        h_nudged, _ = solver.solve(nudged_dynamics, h_free.detach(), x)

    delta = h_nudged - h_free.detach()
    v = - delta / beta

    # Backprop v through f at h_free
    h_at_free = h_free.detach()
    out = model(h_at_free, x)
    out.backward(gradient=v)

    grad_ep = model.W.weight.grad.clone()
    print(f"EP Grad Norm: {grad_ep.norm().item()}")

    sim = F.cosine_similarity(grad_bp.flatten().unsqueeze(0), grad_ep.flatten().unsqueeze(0)).item()
    print(f"Cosine Similarity: {sim:.4f}")

if __name__ == "__main__":
    test_toy_eqprop()
