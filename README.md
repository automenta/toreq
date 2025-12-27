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
- [Adaptive Contingency Framework](#adaptive-contingency-framework)
- [Timeline](#timeline)
- [Adaptive Compute Scaling](#adaptive-compute-scaling)
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

## Adaptive Contingency Framework

This section defines explicit decision criteria for recognizing failure states, identifying novel successes, and pivoting the research direction based on experimental outcomes.

### Failure State Recognition

> [!CAUTION]
> **Complete Failure Criteria** — If ANY of these conditions persist after mitigation attempts, terminate the research direction.

| Failure State | Detection Criteria | Mitigation Attempted | Terminal? |
|---------------|---------------------|----------------------|-----------|
| **Equilibrium Non-Convergence** | >50% of samples fail to converge within 200 iterations | Spectral norm reg, linear attention, reduced α | ✅ Yes |
| **Gradient Mismatch** | Cosine similarity <0.8 at β=0.001 after debugging | Verified equilibrium precision, checked autodiff | ✅ Yes |
| **Catastrophic Slowdown** | Training >500× slower than BP with no accuracy benefit | Anderson accel, early exit, reduced precision | ✅ Yes |
| **Accuracy Collapse** | MNIST accuracy <70% after full hyperparameter sweep | Architecture changes, initialization schemes | ✅ Yes |

**Decision Protocol**:
```
IF gradient_cosine_sim < 0.8 AND linear_attention_tested AND equilibrium_verified:
    → TERMINATE: Publish negative result, document failure mode
    
IF mnist_accuracy < 80% AND hyperparameter_sweep_complete:
    → PIVOT: Investigate hybrid BP+EqProp (use EqProp for specific layers only)
    
IF wall_clock > 200x_BP AND no_accuracy_advantage:
    → TERMINATE: The approach is not practically viable
```

---

### Success Recognition Matrix

> [!TIP]
> **Novel Publishable Outcomes** — Not all successes look like the original hypothesis.

| Outcome | Success Type | Publication Venue | Narrative |
|---------|--------------|-------------------|-----------|
| **Full hypothesis confirmed** | Primary | NeurIPS/ICML main | "EqProp trains transformers with O(1) memory and BP-equivalent gradients" |
| **Linear attention only** | Partial | NeurIPS/ICML main | "EqProp for efficient linear transformers" — still novel, still O(1) memory |
| **Softmax requires hybrid** | Partial | ICLR/TMLR | "Hybrid BP-EqProp: Local learning for attention, global for softmax" |
| **Convergence analysis only** | Theoretical | COLT/ALT | "On the convergence conditions for equilibrium in looped transformers" |
| **Adaptive compute validated** | Emergent | ICML workshop | "Implicit depth: Equilibrium iterations as learned computation budget" |
| **Negative result** | Scientific | NeurIPS track / TMLR | "On the limitations of contrastive Hebbian learning for attention mechanisms" |

**Key Insight**: Even a negative result is publishable if:
1. The hypothesis was reasonable and well-motivated
2. The experiments were rigorous
3. The failure mode is clearly characterized
4. Implications for future work are articulated

---

### Adaptive Pivot Strategies

#### Pivot A: Softmax Attention Fails → Linear Attention Focus

**Trigger**: Gradient mismatch persists with softmax; works with linear attention.

**Action**:
1. Reframe contribution as "TorEqProp for Efficient Transformers"
2. Emphasize that linear attention is an active research area (Performer, Linear Transformers)
3. Drop CIFAR/SST-2, focus on tasks where linear attention is competitive
4. Position as: "Biologically plausible training for the class of efficient transformers"

**Modified Claims**:
- ~~"Train any transformer via EqProp"~~ → "Train linear-attention transformers via EqProp"
- O(1) memory claim remains valid
- Biological plausibility claim remains valid

---

#### Pivot B: Training Too Slow → Focus on Memory Advantage

**Trigger**: Wall-clock is 50-100× slower than BP, but accuracy matches.

**Action**:
1. Reframe as "memory-efficient training for resource-constrained settings"
2. Target edge devices, neuromorphic hardware, federated learning
3. Emphasize that this enables training models that **cannot fit in memory with BP**
4. Add experiments showing TorEqProp trains larger d_model than BP on same GPU

**Modified Claims**:
- Add: "TorEqProp enables training 4× larger models on the same hardware"
- De-emphasize wall-clock; emphasize memory-accuracy tradeoff curve

---

#### Pivot C: Equilibrium Unstable → Analyze Stability Conditions

**Trigger**: Convergence is fragile, requires very specific hyperparameters.

**Action**:
1. Pivot to theoretical contribution: characterize stability conditions
2. Derive precise conditions on attention mechanism for contraction
3. Propose modified attention that guarantees contraction (novel architecture)
4. Paper becomes: "Stable Equilibrium Transformers: Theory and Design"

**Modified Output**:
- New architecture proposal (e.g., "Contractive Attention")
- Theoretical analysis of Jacobian spectral properties
- Practical guidelines for equilibrium-compatible design

---

#### Pivot D: Partial Success → Hybrid Approach

**Trigger**: EqProp works for FFN layers but not attention; or works for early layers but not later ones.

**Action**:
1. Propose "Layerwise Learning Rule Selection"
2. Use EqProp where it works, BP for the rest
3. Still reduces memory (EqProp layers need no activation storage)
4. Frame as: "Toward biologically plausible transformers via hybrid local-global learning"

**Novel Contribution**:
- First systematic study of which layers benefit from local vs. global learning
- Practical hybrid training algorithm
- Analysis of the locality-globality tradeoff in neural network training

---

### Decision Tree

```
                    ┌─────────────────────────────────────┐
                    │  Experiment 1: Gradient Verification │
                    └───────────────────┬─────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            Cosine > 0.99       Cosine 0.8-0.99      Cosine < 0.8
            (Full success)      (Partial)            (Failure)
                    │                   │                   │
                    ▼                   ▼                   ▼
            Continue to          Try linear           Debug deeply
            Experiment 2         attention            (2 weeks max)
                    │                   │                   │
                    │                   │           ┌───────┴───────┐
                    │                   │           ▼               ▼
                    │                   │       Fixed?          Not fixed
                    │                   │           │               │
                    │                   ▼           ▼               ▼
                    │           Linear works?   Continue        TERMINATE
                    │           ┌─────┴─────┐                   Negative
                    │           ▼           ▼                   result paper
                    │         Yes          No
                    │           │           │
                    │           ▼           ▼
                    │     Pivot A:      Pivot C:
                    │     Linear        Stability
                    │     focus         analysis
                    ▼
            ┌───────────────────────────────────────┐
            │  Experiment 2: MNIST Training          │
            └───────────────────┬───────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
        Acc > 95%          Acc 85-95%          Acc < 85%
        Speed < 50×        or Speed > 50×      after sweep
            │                   │                   │
            ▼                   ▼                   ▼
        Full success       Pivot B:             Pivot D:
        → Exp 3           Memory focus          Hybrid
            │               or Pivot D          approach
            ▼
    ┌───────────────────────────────────────────────┐
    │  Experiments 3-4: Scaling & Adaptive Compute   │
    └───────────────────────────────────────────────┘
```

---

### Wall-Clock Reality Check

> [!WARNING]
> **Addressing the Elephant in the Room**: Training speed comparison.

| Method | Forward Passes per Update | Estimated Slowdown |
|--------|---------------------------|-------------------|
| Backprop | 1 forward + 1 backward ≈ 2 | 1× (baseline) |
| TorEqProp | 50 free + 50 nudged = 100 | **50×** (pessimistic) |
| TorEqProp (optimized) | 20 free + 20 nudged = 40 | **20×** (optimistic) |
| TorEqProp + Anderson | 10 free + 10 nudged = 20 | **10×** (aggressive) |

**Honest Assessment**: TorEqProp will likely be 10-50× slower than BP per training step. This must be offset by:

1. **Memory advantage**: Train models that don't fit with BP
2. **Parallelization**: Each equilibrium step is embarrassingly parallel
3. **Hardware co-design**: Neuromorphic chips could run equilibrium natively
4. **Inference benefit**: Adaptive compute at test time

**Paper Strategy**: Acknowledge slowdown upfront; position memory as primary advantage.

---

### Checkpoint Decision Points

| Week | Checkpoint | Go/No-Go Criterion | Pivot If... |
|------|------------|---------------------|-------------|
| 2 | Gradient check | Cosine >0.95 with softmax OR >0.99 with linear | Softmax fails → Pivot A |
| 3 | MNIST baseline | >90% accuracy, <100× slowdown | Accuracy low → Pivot D |
| 4 | MNIST complete | >95% accuracy OR clear pivot narrative | Slowdown high → Pivot B |
| 6 | Scaling | CIFAR >65% OR compelling memory analysis | Neither → focus on theory |
| 8 | Final | Clear publication narrative identified | Always: write the paper |

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

## Adaptive Compute Scaling

TorEqProp is designed to **automatically scale** from commodity hardware to datacenter resources. The research plan adapts based on detected compute tier.

### Hardware Tier Detection

```python
import torch

def detect_compute_tier() -> str:
    """Auto-detect compute tier based on available GPU resources."""
    if not torch.cuda.is_available():
        return "CPU_ONLY"
    
    gpu_count = torch.cuda.device_count()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    
    # Tier classification
    if gpu_count >= 8 or "h100" in gpu_name or "a100" in gpu_name and gpu_count >= 4:
        return "TIER_4_DATACENTER"
    elif "a100" in gpu_name or "a6000" in gpu_name or gpu_mem_gb >= 40:
        return "TIER_3_HIGH_END"
    elif gpu_mem_gb >= 16 or "3090" in gpu_name or "4090" in gpu_name:
        return "TIER_2_PROSUMER"
    elif gpu_mem_gb >= 6:
        return "TIER_1_COMMODITY"
    else:
        return "CPU_ONLY"

# Usage: CONFIG = TIER_CONFIGS[detect_compute_tier()]
```

### Tier Configurations

#### Tier 0: CPU Only (Laptop/Debugging)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 32 | Minimal viable |
| n_heads | 2 | Reduce computation |
| batch_size | 8 | Memory constraint |
| max_iters | 20 | Fast iteration |
| Dataset | MNIST subset (1k) | Quick validation |

**Research scope**: Gradient verification only. ~2 hours per experiment.

---

#### Tier 1: Commodity GPU (6-12GB VRAM)

*Examples: RTX 3060, RTX 4060, GTX 1080 Ti*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 64 | Fits in VRAM |
| n_heads | 4 | Standard ratio |
| batch_size | 32 | Balance speed/memory |
| max_iters | 50 | Full convergence |
| grad_accum_steps | 4 | Simulate larger batch |
| mixed_precision | ✅ fp16 | Essential |
| Dataset | MNIST full | Proof of concept |

**Research scope**: Experiments 1-2 (gradient verification + MNIST training).

**Timeline adjustment**: 
- Week 1-4: Foundation + MNIST
- Scaling experiments deferred to Tier 2+

**Estimated cost**: $0 (local hardware) or ~$50 cloud (spot instances)

---

#### Tier 2: Prosumer GPU (16-24GB VRAM)

*Examples: RTX 3090, RTX 4090, A5000*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 128-256 | Primary validation size |
| n_heads | 4-8 | Flexibility |
| batch_size | 64-128 | Efficient |
| max_iters | 50-100 | Full convergence |
| mixed_precision | ✅ fp16/bf16 | Standard |
| checkpointing | Optional | For larger models |
| Dataset | MNIST, CIFAR-10 | Full validation |

**Research scope**: Experiments 1-3 (gradient verification + training + scaling).

**Timeline**: Full 8-week plan achievable.

**Estimated cost**: $0 (local) or ~$200 cloud

---

#### Tier 3: High-End Workstation (40-80GB VRAM)

*Examples: A100-40GB, A100-80GB, A6000*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 256-512 | Near-publication scale |
| n_heads | 8-16 | Full expressiveness |
| batch_size | 128-256 | Fast iteration |
| max_iters | 100 | High precision |
| parallel_relaxation | ✅ | Batch-parallelized solver |
| Dataset | MNIST, CIFAR-10, SST-2 | Complete benchmark suite |

**Research scope**: All experiments (1-4) + scaling analysis.

**Additional capabilities**:
- Hyperparameter sweeps (Optuna/Ray Tune)
- Multiple random seeds for statistical significance
- Ablation matrix

**Estimated cost**: $400-600 cloud (1-2 weeks A100)

---

#### Tier 4: Datacenter / Multi-GPU

*Examples: 4-8× A100/H100, DGX systems*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 512-1024 | Publication-scale |
| n_heads | 16-32 | Maximum expressiveness |
| batch_size | 512-2048 | Data parallelism |
| max_iters | 100-200 | Precision at scale |
| distributed | ✅ FSDP/DDP | Multi-GPU training |
| Dataset | + WikiText-103, ImageNet-1k | Extended benchmarks |

**Research scope**: Full publication + extension experiments.

**Additional capabilities**:
- Language modeling experiments
- ImageNet classification
- Scaling law analysis (d_model vs. iterations)
- Wall-clock competitive with BP

**Estimated cost**: $1000-3000 cloud

---

### Configuration Presets

```python
TIER_CONFIGS = {
    "CPU_ONLY": {
        "d_model": 32, "n_heads": 2, "d_ff": 128,
        "batch_size": 8, "max_iters": 20, "damping": 0.9,
        "mixed_precision": False, "compile": False,
        "dataset": "mnist_subset", "experiments": [1]
    },
    "TIER_1_COMMODITY": {
        "d_model": 64, "n_heads": 4, "d_ff": 256,
        "batch_size": 32, "max_iters": 50, "damping": 0.9,
        "mixed_precision": True, "compile": True,
        "grad_accum": 4,
        "dataset": "mnist", "experiments": [1, 2]
    },
    "TIER_2_PROSUMER": {
        "d_model": 128, "n_heads": 4, "d_ff": 512,
        "batch_size": 64, "max_iters": 50, "damping": 0.9,
        "mixed_precision": True, "compile": True,
        "dataset": "cifar10", "experiments": [1, 2, 3]
    },
    "TIER_3_HIGH_END": {
        "d_model": 256, "n_heads": 8, "d_ff": 1024,
        "batch_size": 128, "max_iters": 100, "damping": 0.9,
        "mixed_precision": True, "compile": True,
        "parallel_solver": True,
        "dataset": "sst2", "experiments": [1, 2, 3, 4]
    },
    "TIER_4_DATACENTER": {
        "d_model": 512, "n_heads": 16, "d_ff": 2048,
        "batch_size": 512, "max_iters": 100, "damping": 0.9,
        "mixed_precision": True, "compile": True,
        "distributed": True, "parallel_solver": True,
        "dataset": "wikitext103", "experiments": [1, 2, 3, 4, "scaling_laws"]
    }
}
```

### Adaptive Training Script

```python
def main():
    tier = detect_compute_tier()
    config = TIER_CONFIGS[tier]
    
    print(f"🔧 Detected compute tier: {tier}")
    print(f"📊 Model size: d={config['d_model']}, heads={config['n_heads']}")
    print(f"🎯 Experiments enabled: {config['experiments']}")
    
    model = LoopedTransformerBlock(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"]
    )
    
    if config.get("mixed_precision"):
        scaler = torch.cuda.amp.GradScaler()
    
    if config.get("compile") and hasattr(torch, "compile"):
        model = torch.compile(model)
    
    if config.get("distributed"):
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Run applicable experiments
    for exp_id in config["experiments"]:
        run_experiment(exp_id, model, config)
```

### Progressive Research Path

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PROGRESSIVE RESEARCH PATH                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Tier 0 (CPU)     Tier 1          Tier 2          Tier 3    Tier 4 │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Gradient ────────► MNIST ─────────► CIFAR ────────► SST-2 ──────► │
│  Verify             Training         Scaling         Text    LM    │
│                                                             WikiText│
│  [proof of         [MVP paper]      [+ablations]   [full    [scale]│
│   concept]                                          paper]         │
│                                                                     │
│  Deliverable:      Deliverable:     Deliverable:   Deliverable:    │
│  Blog post /       Workshop paper   Conference     Top venue       │
│  Tech report                        submission     submission      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Optimization Strategies

| Strategy | Memory Saving | Speed Cost | When to Use |
|----------|---------------|------------|-------------|
| Mixed precision (fp16) | 40-50% | ~0% | Always on GPU |
| Gradient checkpointing | 60-70% | 20-30% | Large d_model |
| Micro-batching | Proportional | Linear | Tier 1 |
| torch.compile | Varies | -10% (faster) | PyTorch 2.0+ |
| Activation offload | 80%+ | 50-100% | Last resort |

### Scaling Law Experiments (Tier 4)

With datacenter resources, investigate:

1. **Iteration scaling**: How does $T_{converge}$ scale with $d_{model}$?
   - Hypothesis: $T \propto \log(d)$ under proper normalization

2. **β-efficiency scaling**: Optimal β as function of model size
   - Smaller models may tolerate larger β

3. **Memory advantage scaling**: At what $d_{model}$ does O(1) memory dominate?
   - Profile crossover point vs. BP

4. **Wall-clock parity**: When does TorEqProp match BP throughput?
   - Critical for practical adoption

---

## Software Dependencies

```
torch >= 2.0
einops
wandb
scipy (for Anderson acceleration)
optuna (optional, hyperparameter search)
```

### Personnel

| Tier | Researcher Time | Notes |
|------|-----------------|-------|
| Tier 0-1 | Part-time (evenings/weekends) | Hobby project viable |
| Tier 2 | 4 weeks full-time | MVP paper |
| Tier 3-4 | 8 weeks full-time + advisor | Full publication |

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