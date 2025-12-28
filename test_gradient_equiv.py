import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver

def test_gradient_equivalence():
    print("Testing Gradient Equivalence...")

    # Setup
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Model configs
    d_model = 64
    n_heads = 4
    d_ff = 256
    batch_size = 10
    seq_len = 5
    n_classes = 10

    # Using Linear Attention and Symmetric FFN for energy based dynamics
    model = LoopedTransformerBlock(d_model, n_heads, d_ff, use_linear_attn=True, symmetric=True).to(device)
    output_head = nn.Linear(d_model, n_classes).to(device)

    # Inputs
    x = torch.randn(seq_len, batch_size, d_model, device=device)
    y = torch.randint(0, n_classes, (batch_size,), device=device)

    solver = EquilibriumSolver(max_iters=50, tol=1e-5, damping=0.9)
    beta = 0.001

    # 1. Free Phase
    with torch.no_grad():
        h0 = torch.zeros_like(x)
        h_free, iters_free = solver.solve(model, h0, x)
    print(f"Free phase converged in {iters_free} iterations.")

    # 2. BP Gradient
    model.zero_grad()
    output_head.zero_grad()

    h_bp = torch.zeros_like(x, requires_grad=True)
    h = torch.zeros_like(x)
    for _ in range(50):
        h = (1 - 0.9) * h + 0.9 * model(h, x)

    y_pred = output_head(h.mean(dim=0))
    loss_bp = F.cross_entropy(y_pred, y)
    loss_bp.backward()

    grad_bp = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_bp[name] = param.grad.clone()

    print(f"BP Loss: {loss_bp.item()}")

    # 3. EqProp Gradient
    model.zero_grad()
    output_head.zero_grad()

    h_free_detached = h_free.detach()
    h_free_detached.requires_grad_(True)

    def nudged_dynamics(h, x):
        with torch.enable_grad():
            h_in = h.detach().requires_grad_(True)
            h_out = model(h_in, x)
            y_p = output_head(h_out.mean(dim=0))
            l = F.cross_entropy(y_p, y)
            g = torch.autograd.grad(l, h_in)[0]
        return h_out - beta * g

    with torch.no_grad():
        h_nudged, iters_nudged = solver.solve(nudged_dynamics, h_free_detached, x)
    print(f"Nudged phase converged in {iters_nudged} iterations.")

    # Update: v = (h_nudged - h_free) / beta
    delta = h_nudged - h_free_detached
    v = delta / beta

    model.zero_grad()
    h_at_free = h_free.detach()
    out_free = model(h_at_free, x)
    out_free.backward(gradient=v)

    grad_ep = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_ep[name] = param.grad.clone()

    # Compare
    cos_sims = []
    for name in grad_bp:
        if name in grad_ep:
            g_bp = grad_bp[name].flatten()
            g_ep = grad_ep[name].flatten()
            if g_bp.numel() > 0:
                sim = F.cosine_similarity(g_bp.unsqueeze(0), g_ep.unsqueeze(0)).item()
                cos_sims.append(sim)
                print(f"{name}: Cosine Sim = {sim:.4f}")

    if cos_sims:
        avg_sim = sum(cos_sims) / len(cos_sims)
        print(f"Average Cosine Similarity: {avg_sim:.4f}")
    else:
        print("No gradients to compare.")

if __name__ == "__main__":
    test_gradient_equivalence()
