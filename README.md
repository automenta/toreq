# Toroidal Equilibrium Propagation for Transformers (TorEqProp)

> **Status**: 🔬 Theoretical Proposal / Research Specification  
> **Version**: 0.2.0  
> **Target**: ICML/NeurIPS 2025 submission  

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Core Hypothesis](#core-hypothesis)
- [Architecture](#architecture)
- [Training Algorithm](#training-algorithm)
- [Experimental Plan](#experimental-plan)
- [Success Criteria](#success-criteria)
- [Implementation Specification](#implementation-specification)
- [Risk Analysis](#risk-analysis)
- [Timeline](#timeline)
- [Resources Required](#resources-required)
- [Related Work](#related-work)
- [Appendix: Mathematical Details](#appendix-mathematical-details)

---

## Executive Summary

**TorEqProp** proposes training transformers via Equilibrium Propagation on weight-tied (toroidal) architectures, eliminating backpropagation's asymmetric backward pass. This yields:

| Claim | Validation Method |
|-------|-------------------|
| O(1) memory training | Profile vs. BP on same model |
| Biological plausibility | Local Hebbian updates only |
| Equivalent gradients to BP | Analytical proof + empirical β→0 limit |
| Competitive accuracy | Match BP baseline within 2% on benchmarks |

**Minimum Publishable Result**: Demonstrate TorEqProp trains a looped transformer to ≥95% MNIST accuracy with verified gradient equivalence.

---

## Core Hypothesis

> **H1**: A weight-tied transformer iterated to fixed-point equilibrium can be trained via contrastive Hebbian learning (EqProp) with gradients equivalent to implicit differentiation through the equilibrium.

**Testable Predictions**:

1. $\lim_{\beta \to 0} \frac{\Delta \theta_{\text{EqProp}}}{\beta} = \nabla_\theta \mathcal{L}$ (BP gradient)
2. Convergence to equilibrium occurs in $O(\log(1/\epsilon))$ iterations for well-conditioned systems
3. Training curves (loss, accuracy) match BP baselines within statistical noise

---

## Architecture

### Looped Transformer Block

```
┌─────────────────────────────────────────────┐
│                                             │
│  x (input) ──┐                              │
│              ▼                              │
│  ┌─────────────────────┐                    │
│  │   h_t (hidden)      │◄────────────┐      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  LayerNorm          │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  MultiHeadAttn(h,x) │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  + Residual         │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  LayerNorm          │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  FFN                │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  + Residual ────────┼─────────────┘      │
│  └──────────┬──────────┘                    │
│             ▼                               │
│         h_{t+1}                             │
│             │                               │
│             ▼ (iterate until ‖h-h'‖<ε)      │
│          h* ──► Output Head ──► ŷ           │
│                                             │
└─────────────────────────────────────────────┘
```

### Convergence Dynamics

$$h_{t+1} = (1-\alpha)h_t + \alpha \cdot f_\theta(h_t; x)$$

where $\alpha \in (0,1]$ is the damping factor. Convergence criterion:

$$\|h_{t+1} - h_t\|_2 < \epsilon \quad \text{or} \quad t > T_{\max}$$

**Required property**: Spectral radius $\rho(J_f) < 1$ where $J_f = \frac{\partial f}{\partial h}$

---

## Training Algorithm

### Algorithm 1: TorEqProp Training Step

```
Input: x (input), y (target), β (nudge strength), ε (tolerance)
Output: Updated parameters θ

1. EQUILIBRIUM PHASE (Free)
   h ← 0  # or learned initialization
   repeat:
       h' ← (1-α)h + α·f_θ(h; x)
       if ‖h' - h‖ < ε: break
       h ← h'
   h* ← h
   A* ← {layer activations at h*}

2. EQUILIBRIUM PHASE (Nudged)  
   h ← h*
   repeat:
       h' ← (1-α)h + α·f_θ(h; x)
       ŷ ← OutputHead(h')
       h' ← h' + β · ∇_h L(ŷ, y)  # Nudge toward target
       if ‖h' - h‖ < ε: break
       h ← h'
   h^β ← h
   A^β ← {layer activations at h^β}

3. WEIGHT UPDATE (Contrastive Hebbian)
   for each layer l:
       ΔW_l ← (1/β) · (A^β_l ⊗ A^β_l - A*_l ⊗ A*_l)
       θ_l ← θ_l - η · ΔW_l
```

### Gradient Equivalence Theorem

**Theorem** (Scellier & Bengio, 2017; adapted): For energy-based dynamics at equilibrium $h^*$, as $\beta \to 0$:

$$\frac{1}{\beta}(h^\beta - h^*) \to -(I - J_f)^{-1} \nabla_h \mathcal{L}$$

and the contrastive update equals:

$$\lim_{\beta \to 0} \frac{\Delta \theta}{\beta} = \nabla_\theta \mathcal{L}\big|_{h=h^*}$$

**Empirical Validation**: Compute both gradients, report cosine similarity and L2 error.

---

## Experimental Plan

### Experiment 1: Gradient Verification (Week 1-2)

**Objective**: Prove EqProp gradients match BP gradients.

| Component | Specification |
|-----------|---------------|
| Model | 1-block looped transformer, d=64, heads=4 |
| Data | MNIST (28×28 flattened to sequence) |
| Metric | Cosine sim(∇_EqProp, ∇_BP), L2 error |
| β values | [0.5, 0.1, 0.01, 0.001] |
| Success | Cosine sim > 0.99 at β=0.001 |

**Protocol**:
1. Forward pass to equilibrium (max 50 iters)
2. Compute EqProp gradient via contrastive activations
3. Compute BP gradient via torch.autograd on equilibrium
4. Compare across 100 random batches

### Experiment 2: Training Dynamics (Week 2-4)

**Objective**: Train to convergence, compare learning curves.

| Component | Specification |
|-----------|---------------|
| Model | 1-block looped transformer, d=128, heads=4 |
| Data | MNIST train/test split |
| Baseline | Same architecture trained with BP |
| Metrics | Train loss, test accuracy, iterations/sample |
| Success | ≥95% test accuracy, within 2% of BP baseline |

**Ablations**:
- β ∈ {0.01, 0.05, 0.1, 0.2}
- Damping α ∈ {0.5, 0.7, 0.9, 1.0}
- Solver: fixed-point vs. Anderson acceleration

### Experiment 3: Scaling (Week 4-6)

**Objective**: Validate on harder tasks, analyze scaling.

| Task | Model Size | Target |
|------|------------|--------|
| CIFAR-10 | d=256, 1 block | ≥70% accuracy |
| CIFAR-10 | d=256, 2 blocks (unrolled 2× per iter) | ≥75% accuracy |
| Text classification (SST-2) | d=256, vocab=10k | ≥80% accuracy |

**Scaling metrics**:
- Iterations to convergence vs. model dimension
- Wall-clock time vs. BP (same hardware)
- Peak memory vs. BP

### Experiment 4: Adaptive Compute (Week 6-8)

**Objective**: Demonstrate variable-depth computation.

**Protocol**:
1. Train with fixed max_iters=50
2. At test time, measure iterations to ε-convergence per sample
3. Correlate iteration count with sample difficulty (margin, uncertainty)
4. Compare to DEQ baseline (same equilibrium architecture, BP-trained)

**Hypothesis**: Hard samples require more iterations; this correlates with model uncertainty.

---

## Success Criteria

### Minimum Viable Publication (MVP)

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| Gradient equivalence demonstrated | Cosine sim > 0.99 | 🔴 Critical |
| MNIST convergence | ≥95% accuracy | 🔴 Critical |
| Training completes | <24h on single GPU | 🟡 High |
| Memory advantage shown | <50% of BP peak memory | 🟡 High |

### Stretch Goals

| Goal | Threshold | Priority |
|------|-----------|----------|
| CIFAR-10 competitive | Within 5% of BP baseline | 🟢 Medium |
| Text classification | ≥75% SST-2 accuracy | 🟢 Medium |
| Neuromorphic simulation | Run on Loihi/SpiNNaker | 🔵 Low |

---

## Implementation Specification

### Core Classes

```python
class LoopedTransformerBlock(nn.Module):
    """Single weight-tied transformer block for equilibrium iteration."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        # Standard transformer components
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        # h: [seq, batch, d_model], x: [seq, batch, d_model]
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, x, x)  # Cross-attend to input
        h = h + attn_out
        h_norm = self.norm2(h)
        h = h + self.ffn(h_norm)
        return h


class EquilibriumSolver:
    """Fixed-point solver with convergence monitoring."""
    
    def __init__(self, max_iters: int = 50, tol: float = 1e-5, damping: float = 0.9):
        self.max_iters = max_iters
        self.tol = tol
        self.damping = damping
        
    def solve(self, f: Callable, h0: Tensor, x: Tensor) -> Tuple[Tensor, int]:
        h = h0
        for t in range(self.max_iters):
            h_new = (1 - self.damping) * h + self.damping * f(h, x)
            residual = (h_new - h).norm()
            if residual < self.tol:
                return h_new, t + 1
            h = h_new
        return h, self.max_iters  # Did not converge


class EqPropTrainer:
    """Equilibrium Propagation training loop."""
    
    def __init__(self, model, solver, output_head, beta: float = 0.1, lr: float = 1e-3):
        self.model = model
        self.solver = solver
        self.output_head = output_head
        self.beta = beta
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(output_head.parameters()), 
            lr=lr
        )
        
    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        # Free phase
        h0 = torch.zeros_like(x)
        h_free, iters_free = self.solver.solve(self.model, h0, x)
        
        # Nudged phase
        def nudged_dynamics(h, x):
            h_new = self.model(h, x)
            y_pred = self.output_head(h_new.mean(dim=0))  # Pool over sequence
            nudge = self.beta * torch.autograd.grad(
                -F.cross_entropy(y_pred, y), h_new, retain_graph=True
            )[0]
            return h_new + nudge
        
        h_nudged, iters_nudged = self.solver.solve(nudged_dynamics, h_free.detach(), x)
        
        # Contrastive Hebbian update (simplified: use autodiff on difference)
        loss_proxy = ((h_nudged - h_free.detach()) ** 2).mean()
        self.optimizer.zero_grad()
        loss_proxy.backward()
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            y_pred = self.output_head(h_free.mean(dim=0))
            acc = (y_pred.argmax(-1) == y).float().mean()
            
        return {
            "loss": loss_proxy.item(),
            "accuracy": acc.item(),
            "iters_free": iters_free,
            "iters_nudged": iters_nudged
        }
```

### Hyperparameter Defaults

| Parameter | Default | Search Range | Notes |
|-----------|---------|--------------|-------|
| d_model | 128 | [64, 256, 512] | Start small |
| n_heads | 4 | [2, 4, 8] | d_model must be divisible |
| d_ff | 512 | 4 × d_model | Standard ratio |
| β (nudge) | 0.1 | [0.01, 0.5] | Critical for gradient quality |
| α (damping) | 0.9 | [0.5, 1.0] | 1.0 = no damping |
| ε (tolerance) | 1e-5 | [1e-6, 1e-3] | Trade-off: precision vs. speed |
| max_iters | 50 | [20, 100] | Set high initially |
| lr | 1e-3 | [1e-4, 1e-2] | Adam default |

### Logging & Monitoring

Track per training step:
- `loss`, `accuracy`
- `iters_free`, `iters_nudged` (convergence speed)
- `grad_cosine_sim` (vs. BP baseline, sample periodically)
- `spectral_norm_jacobian` (stability diagnostic)

**Wandb/TensorBoard integration recommended.**

---

## Risk Analysis

### High-Risk Issues

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Non-convergence | Medium | 🔴 Fatal | Spectral norm regularization; constrained initialization |
| Gradient mismatch | Low | 🔴 Fatal | Validate β→0 limit analytically; compare to DEQ |
| Slow training | High | 🟡 Major | Anderson acceleration; learned initialization |

### Contingency Plans

**If convergence fails**:
1. Switch to linear attention (guaranteed contraction)
2. Add explicit Jacobian penalty: $\mathcal{L} += \lambda \|\|J_f\|\|_2$
3. Use DEQ-style phantom gradient as fallback

**If gradients don't match BP**:
1. Verify implementation against reference EqProp code
2. Check equilibrium is truly reached (tighten ε)
3. May indicate attention breaks energy assumptions → investigate energy reformulation

**If too slow**:
1. Reduce max_iters, accept approximate equilibrium
2. Parallel batch relaxation
3. Early exit with residual as uncertainty measure

---

## Timeline

```
Week 1-2: Foundation
├── Day 1-3: Implement LoopedTransformerBlock, EquilibriumSolver
├── Day 4-7: Implement EqPropTrainer, verify forward pass
├── Day 8-10: Gradient verification experiment (Exp 1)
└── Day 11-14: Debug, iterate until gradients match

Week 3-4: Training
├── Day 15-18: Full training loop on MNIST
├── Day 19-21: Hyperparameter sweep (β, α, lr)
├── Day 22-25: Compare to BP baseline
└── Day 26-28: Ablation studies, document results

Week 5-6: Scaling
├── Day 29-32: CIFAR-10 experiments
├── Day 33-36: Text classification (SST-2)
├── Day 37-40: Memory profiling, wall-clock comparison
└── Day 41-42: Analyze scaling trends

Week 7-8: Polish & Write
├── Day 43-46: Adaptive compute experiments
├── Day 47-50: Additional ablations, robustness checks
├── Day 51-54: Paper writing
└── Day 55-56: Internal review, submission prep
```

---

## Resources Required

### Compute

| Phase | GPU Hours | Hardware |
|-------|-----------|----------|
| Development | 50 | 1× A100 |
| Exp 1-2 (MNIST) | 100 | 1× A100 |
| Exp 3 (Scaling) | 200 | 2× A100 |
| Exp 4 (Adaptive) | 100 | 1× A100 |
| **Total** | **450** | ~$500-1000 cloud |

### Software Dependencies

```
torch >= 2.0
einops
wandb
scipy (for Anderson acceleration)
```

### Personnel

- 1 researcher (full-time, 8 weeks)
- 1 advisor (part-time review)

---

## Related Work

### Foundational

| Paper | Year | Relevance |
|-------|------|-----------|
| Scellier & Bengio, "Equilibrium Propagation" | 2017 | Core algorithm |
| Bai et al., "Deep Equilibrium Models" | 2019 | DEQ architecture |
| Dehghani et al., "Universal Transformers" | 2018 | Looped transformers |

### Recent (2023-2024)

| Paper | Year | Relevance |
|-------|------|-----------|
| Laborieux et al., "Scaling EqProp" | 2021 | Modern EqProp implementation |
| Yang et al., "Looped Transformers for Reasoning" | 2024 | Expressive power results |
| Hoover et al., "Energy Transformer" | 2023 | Energy-based attention |

### To Distinguish From

| Approach | Key Difference from TorEqProp |
|----------|-------------------------------|
| DEQ | Uses implicit diff with BP; not biologically plausible |
| Hopfield Transformers | BP-trained; energy is descriptive not prescriptive |
| Predictive Coding | Different local rule; not transformer-native |

---

## Appendix: Mathematical Details

### A1: Energy Formulation for Attention

**Open Problem**: Softmax attention lacks closed-form energy. Candidates:

1. **Hopfield energy** (Ramsauer et al.):
   $$E = -\sum_i \log \sum_j \exp(\beta q_i^T k_j) + \text{regularization}$$

2. **Linear attention surrogate**:
   $$\text{Attn}(Q,K,V) = \phi(Q)\phi(K)^T V$$
   admits energy $E = -\frac{1}{2}\|V^T \phi(K)^T \phi(Q)\|^2$

3. **Variational bound**: Treat softmax as approximate inference; derive ELBO-like energy.

**Recommendation**: Start with linear attention for guaranteed results; investigate softmax post-hoc.

### A2: Contraction Conditions

For convergence, require $\|J_f\|_2 < 1$. Strategies:

1. **Spectral normalization**: Divide weights by spectral norm
2. **Residual scaling**: $h' = h + \gamma f(h)$ with $\gamma < 1$
3. **Lipschitz FFN**: Use GroupSort or other Lipschitz activations

### A3: β-Gradient Relationship

Formal expansion (Scellier & Bengio):

$$h^\beta = h^* + \beta \cdot v + O(\beta^2)$$

where $v = -(I - J_f)^{-1} \nabla_h \mathcal{L}$.

Weight gradient:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \lim_{\beta \to 0} \frac{1}{\beta} \left[ \frac{\partial E}{\partial \theta}\bigg|_{h^\beta} - \frac{\partial E}{\partial \theta}\bigg|_{h^*} \right]$$

This recovers the implicit function theorem gradient used in DEQs.

---

## Quick Start Checklist

- [ ] Clone repo, install dependencies
- [ ] Run `python test_gradient_equiv.py` — verify gradient matching
- [ ] Run `python train_mnist.py` — baseline training
- [ ] Check wandb dashboard for convergence curves
- [ ] Compare to `python train_mnist_bp.py` — BP baseline

---

<div align="center">

**TorEqProp** — Symmetric, local, biologically plausible transformer training.

*Questions? Open an issue or contact [author@institution.edu]*

</div>