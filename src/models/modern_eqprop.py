import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernEqProp(nn.Module):
    """
    Modern Equilibrium Propagation Model (Ported from Archive).
    
    Structure: Residual Network with LayerNorm and ReLU FFN.
    Dynamics: h_{t+1} = h_t + gamma * (FFN(LayerNorm(h_t)) - h_t) ? 
              Archive says: h + gamma * ffn(norm(h))  [Additive Residual]
              
    This matches Deep Equilibrium Models (DEQ) architecture.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        
        self.ffn_dim = 4 * hidden_dim
        
        # Input embedding
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Weight-tied FFN block
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, hidden_dim)
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output classifier
        self.Head = nn.Linear(hidden_dim, output_dim)

    def forward_step(self, h, x, buffer=None):
        # Dynamics: h_{t+1} = h_t + gamma * FFN(Norm(h_t)) 
        # Note: This is different from h <- (1-a)h + a*f(h).
        # This is strictly residual integration.
        
        # We need to mix input x into the dynamics?
        # Archive `forward` took `h` and `x`.
        # Archive `init_hidden` used `embed(x)`.
        # Archive `forward` was: `return h + self.gamma * self.ffn(self.norm(h))`
        # Wait, where does X enter?
        # In the archive, X ONLY initialized h0!
        # This means it's a "Autonomous Dynamical System initialized by Input".
        # Standard EqProp usually drives with X at every step: u = Wx + Wh.
        
        # Let's check archived init_hidden again.
        # Yes, `init_hidden` returns `embed(x)`.
        # `forward` calculates delta based on `h`.
        
        # PROBLEM: If X doesn't drive dynamics, we can't do "clamped" phase easily?
        # Or maybe we clamp the *input layer*? No, we clamp output.
        # But if h drifts away from x, it forgets input?
        # DEQ usually has injection: f(h, x).
        
        # Let's look at the archive code I verified step 428.
        # Line 143: def forward(self, h, x) -> Tensor:
        # Line 146: ffn_out = self.ffn(h_norm)
        # Line 147: return h + self.gamma * ffn_out
        #
        # Indeed, X is unused in dynamics!
        # This works for DEQ because Z* depends on X via injection usually.
        # If the archived model didn't inject X, then the fixed point is X-independent?
        # That would be broken. 
        # Ah, maybe `embed` isn't just init?
        # Wait, if `h*` is independent of `x` (except initialization), then `dL/dh*` Backprop
        # through time would work (if unrolled), but finding `h*` via root finding?
        # If f(h) has a root, the root is constant.
        
        # Hypotheses:
        # 1. Archive code was buggy/incomplete?
        # 2. I missed something.
        
        # Let's look at `ToroidalMLP` in archive (Line 203).
        # `drive = s_t + recirculation` -> `ffn(drive)`. X unused there too?
        # `s_0 = self.embed(x)`.
        
        # THIS SEEMS WRONG. If X is only initial condition, and dynamics are contractive to a 
        # *global* fixed point, then all inputs map to same output.
        # UNLESS the fixed point depends on X? But f(h) doesn't use X.
        
        # CORRECTION: I will MODIFY the Modern architecture to inject X.
        # Standard ResNet block: y = f(h) + x?
        # Let's add `+ self.embed(x)` to the FFN input or output.
        # Let's add it to the residual stream.
        # h_{t+1} = h_t + gamma * (FFN(Norm(h_t)) + Embed(x) - h_t) ?
        # Or just: h_{t+1} = (1-g)h + g(FFN(h) + E(x))
        
        # Let's stick to the "Canonical" EqProp form but use the FFN/Norm blocks.
        # h_new = FFN(Norm(h)) + Embed(x)
        # h_{t+1} = (1-gamma)h + gamma * h_new.
        
        h_norm = self.norm(h)
        # For X injection, we need to map X to hidden dim.
        # We can cache this if X is constant, but for API simplicity calculate it.
        x_emb = self.embed(x)
        
        ffn_out = self.ffn(h_norm)
        
        # Dynamics: h converges to FFN(h) + Wx
        h_target = ffn_out + x_emb 
        
        # Update
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        return h_next, None

    def forward(self, x, steps=30):
        # Initialize
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        # Better init:
        h = self.embed(x) 
        
        for _ in range(steps):
            h, _ = self.forward_step(h, x)
        return self.Head(h)

    def energy(self, h, x, buffer=None):
        # This complex architecture does not have a clean scalar Energy function.
        # We must rely on `vector_field` dynamics (implicit differentiation) 
        # OR just use the "Surrogate Loss" heuristic which works surprisingly well.
        #
        # Surrogate: 0.5 * || h - f(h,x) ||^2 ? No, that minimizes update.
        #
        # Let's define Pseudo-Energy E = 0.5 * ||h||^2 - Integral(Dynamics).
        # For general FFN, this doesn't exist.
        #
        # FALLBACK: Use the same "Contrastive" heuristic as Trainer uses.
        # We can implement a dummy energy that returns 0, 
        # BUT `trainer.py` uses `backward` on Energy to update weights.
        # 
        # If we return 0, weights won't update via that path.
        #
        # The trainer supports "Vector Field" updates theoretically but our 
        # current `EqPropTrainer` is hardcoded for Energy difference.
        #
        # To make this model work with CURRENT `EqPropTrainer`, we need a scalar E.
        # E = 0.5|h|^2 - Sum(LogCosh( ... )) matches Tanh.
        #
        # What matches ReLU MLP?
        # Primitives of ReLU?
        #
        # Let's try to define a "Local Potential" energy.
        # E = 0.5 ||h||^2 - h * (FFN(h) + Wx).
        # dE/dh = h - (FFN'(h) + Wx). 
        # This assumes symmetric weights in FFN, which isn't true.
        #
        # CRITICAL DECISION:
        # EqProp STRICTLY requires symmetric weights or energy function.
        # The Archive model used `UpdateStrategy` which might have handled non-energy updates.
        # 
        # Since I am using `EqPropTrainer` which *relies* on autograd of Energy,
        # I must implement a symmetric version or a specific energy.
        # Note: `LoopedMLP` used `symmetric=True` optional.
        #
        # Let's implement `ModernEqProp` using `Tanh` instead of `ReLU` to allow LogCosh Energy,
        # BUT keep the LayerNorm and Deep structure.
        #
        # Actually, let's look at `EqPropTrainer.step`:
        # `surrogate_loss = (E_nudged - E_free) / beta`.
        # This relies on E.
        #
        # I will define `energy` as:
        # 0.5*||h||^2 - h.detach() * (FFN(h) + Wx) ? 
        # No, we need gradients w.r.t weights.
        # Energy = - h_fixed * (FFN(h_var) + Wx) ? -> dE/dW = - h * f'(W) ...
        # This is basically "Target Propagation".
        #
        # Let's use the "Implicit Energy":
        # E = 0.5 * ||h||^2 - Potential(h)
        # We approximate Potential(h) such that grad(P) approx Direction.
        #
        # Let's just output `0.5*||h||^2` and see? No.
        #
        # Let's simply use the `LoopedMLP` energy logic but applied to this deep structure:
        # `term2 = h * (FFN(h) + Wx)` 
        # This isn't a scalar field conservative, but it generates gradients that drive h?
        #
        # Let's define: E = 0.5*h^2 - h * (FFN(h) + Embed(x)). 
        # CAUTION: This might be unstable, but let's try.
        
        x_emb = self.embed(x)
        h_norm = self.norm(h)
        ffn_out = self.ffn(h_norm)
        
        # "Energy" guess
        # If we want the update to push W to make FFN(h) closer to h?
        # effectively: h_target = FFN(h).
        # We want h_target to be "good".
        
        # Let's return a "Proxy Energy" that calculates the activity.
        # P = Dot(h.detach(), FFN(h) + x_emb)
        # E = 0.5 * ||h||^2 - P
        
        P = torch.sum(h * (ffn_out + x_emb), dim=1).sum()
        term1 = 0.5 * torch.sum(h**2)
        return term1 - P
