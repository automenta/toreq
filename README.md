# Toroidal Equilibrium Propagation for Transformers (TorEqProp)

> **Status**: 🔬 Theoretical Proposal / Research Specification  
> **Version**: 0.1.0  
> **License**: MIT  

---

## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Architecture: Looped/Toroidal Transformer](#architecture-loopedtoroidal-transformer)
- [Training: Equilibrium Propagation on the Torus](#training-equilibrium-propagation-on-the-torus)
- [Energy Formulation](#energy-formulation)
- [Advantages](#advantages)
- [Comparison with Existing Approaches](#comparison-with-existing-approaches)
- [Implementation Guidelines](#implementation-guidelines)
- [Pseudocode](#pseudocode)
- [Potential Challenges & Mitigations](#potential-challenges--mitigations)
- [Open Research Questions](#open-research-questions)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Related Work](#related-work)
- [Citation](#citation)

---

## Overview

**Toroidal Equilibrium Propagation (TorEqProp)** is a novel training paradigm that combines **looped (or toroidal/recurrent-depth) transformers** with **Equilibrium Propagation (EqProp)** to achieve fully symmetric, biologically plausible credit assignment.

By "looping the transformer into a torus" — recursively feeding outputs back as inputs with weight-tying — the feedforward transformer becomes a convergent recurrent network capable of relaxing to fixed-point equilibria. EqProp then trains this looped architecture using only forward-phase relaxations (free and nudged equilibria), eliminating the asymmetric backward pass of backpropagation (BP).

This approach directly addresses a fundamental intuition: **equalizing forward (inference) and backward (learning) signal metrics through identical recurrent dynamics**. Both phases use the same looped forward computations, ensuring balanced propagation of activations and gradients.

> [!IMPORTANT]
> This document is a **theoretical research specification**, not an implemented system. It provides a complete blueprint for researching and implementing TorEqProp as a genuine post-BP paradigm.

---

## Key Innovations

The novelty of TorEqProp lies in the **direct application of EqProp's contrastive Hebbian mechanism to toroidal transformers**, yielding:

1. **Scalable, local, hardware-friendly alternative** to BP for transformer-scale models
2. **Distinct from existing approaches**:
   - Unlike energy-based transformers → no implicit differentiation required
   - Unlike DEQs → no root-finding with BP-like gradients
3. **Symmetric dynamics** — identical compute paths for inference and learning

---

## Architecture: Looped/Toroidal Transformer

### Base Structure

Start with a standard transformer block $f_\theta(h; x)$, where:
- $h$ — hidden state
- $x$ — input sequence (with positional encodings if needed)

### Toroidal Looping

Recursively apply the same block with weight-tying:

$$h_{t+1} = h_t + f_\theta(h_t; x) \quad \text{(residual form for stability)}$$

or simply:

$$h_{t+1} = f_\theta(h_t; x)$$

This creates a **weight-tied recurrent network with toroidal topology** (closed loop, no start/end).

### Inference (Free Phase)

Iterate until convergence to equilibrium $h^*$ where:

$$h^* \approx f_\theta(h^*; x) \quad \text{or residual form:} \quad 0 \approx f_\theta(h^*; x)$$

**Convergence methods**:
- Fixed-point iteration with damping
- Broyden's method (fast, low-memory)
- Anderson acceleration

**Adaptive compute**: More iterations for complex inputs → System 2-like "thinking"

The equilibrium $h^*$ represents the final processed representation/prediction.

> [!NOTE]
> This is inspired by looped transformers (e.g., Universal Transformers, recent 2024–2025 works on expressive power and reasoning) but optimized for **convergence** rather than finite unrolling.

---

## Training: Equilibrium Propagation on the Torus

EqProp trains energy-based models via two symmetric relaxation phases:

### Phase 1: Free Phase
Relax to equilibrium $h^*$ with input $x$ clamped (no target nudge). This minimizes an implicit energy $E(h; \theta)$.

### Phase 2: Nudged Phase
Slightly perturb output neurons toward the target $y$ with small factor $\beta > 0$:

$$\text{Add } \beta(y - h_L) \text{ to final layer dynamics}$$

where $h_L$ is the output projection. Relax to nearby equilibrium $h^\beta$.

### Phase 3: Weight Update
Apply the **contrastive Hebbian rule**:

$$\Delta \theta \propto \left( h^\beta (h^\beta)^T - h^* (h^*)^T \right)$$

(or per-synapse co-activations — local and Hebbian)

### Key Properties

| Property | Description |
|----------|-------------|
| **BP Equivalence** | In the limit $\beta \to 0$, exactly matches BP gradients for the fixed-point system |
| **Symmetric Dynamics** | Both phases use identical toroidal dynamics — fully equalized forward/backward signals |
| **No Backward Pass** | Eliminates weight transport problem; highly local updates |

---

## Energy Formulation

> [!TIP]
> Optional but recommended for theoretical grounding

Frame the looped transformer as minimizing an energy:

$$E(h; \theta, x) = -\frac{1}{2} h^T W h + \phi(h) + \psi(h, x)$$

where:
- $W$ — learned weight matrix
- $\phi(h)$ — non-linear potential (e.g., from activation functions)
- $\psi(h, x)$ — input coupling term

**Dynamics**:

$$\dot{h} = -\frac{\partial E}{\partial h}$$

This leads to Hopfield-like convergence guarantees under appropriate conditions.

> [!WARNING]
> The simplified energy $E = -\frac{1}{2} h^T W h$ is illustrative. A rigorous energy formulation for the full transformer block (including attention) remains an open research question. See [Open Research Questions](#open-research-questions).

---

## Advantages

| Advantage | Description |
|-----------|-------------|
| **Biological Plausibility** | Symmetric dynamics, local Hebbian updates, no asymmetric BP |
| **Memory Efficiency** | O(1) memory (no computational graph unrolling) |
| **Adaptive Compute** | Variable iterations at inference time based on input complexity |
| **Scalability** | Leverages transformer's expressiveness; potential for better reasoning via deeper equilibria |
| **Hardware-Friendly** | Suitable for neuromorphic chips (symmetric recurrent relaxations) |
| **Uncertainty Estimation** | Equilibria enable natural uncertainty quantification and iterative refinement |

---

## Comparison with Existing Approaches

| Approach | Gradient Method | Memory | Biological Plausibility | Attention Native |
|----------|-----------------|--------|-------------------------|------------------|
| **Standard BP** | Backprop | O(L) | ❌ Low | ✅ Yes |
| **DEQ** | Implicit diff + BP | O(1) | ❌ Low | ✅ Yes |
| **Hopfield Transformers** | BP on energy | O(L) | ⚠️ Partial | ⚠️ Modified |
| **Predictive Coding** | Local updates | O(1) | ✅ High | ❌ No |
| **TorEqProp (proposed)** | Contrastive Hebbian | O(1) | ✅ High | ⚠️ TBD |

---

## Implementation Guidelines

### 1. Base Block

Use a standard transformer layer with modifications for convergence:

```
TransformerBlock(h, x):
    h_norm = LayerNorm(h)
    attn_out = MultiHeadAttention(h_norm, context=x)
    h = h + attn_out                    # Residual
    h_norm = LayerNorm(h)
    ffn_out = FeedForward(h_norm)
    h = h + ffn_out                     # Residual
    return h
```

**Key considerations**:
- Residual connections aid convergence
- LayerNorm stabilizes dynamics
- Consider spectral normalization on weights

### 2. Relaxation Solver

| Method | Speed | Memory | Stability |
|--------|-------|--------|-----------|
| Fixed-point iteration | Slow | O(1) | High |
| Broyden's method | Fast | O(k·d) | Medium |
| Anderson acceleration | Fast | O(k·d) | Medium |

Recommended: **Broyden's method** with k=5 history vectors.

### 3. Nudging Strategy

For different task types:

| Task | Nudging Approach |
|------|------------------|
| Classification | Nudge classifier head on $h^*$ toward one-hot target |
| Sequence-to-sequence | Nudge final tokens toward target tokens |
| Language modeling | Nudge next-token prediction head |

### 4. Stabilization Techniques

- **Jacobian regularization**: Penalize $\|J_{f_\theta}\|$ to ensure contraction
- **Timestep encodings**: From looped transformer literature
- **Damping factor**: $h_{t+1} = (1-\alpha)h_t + \alpha f_\theta(h_t; x)$ with $\alpha < 1$

### 5. Suggested Task Progression

1. **MNIST** — Simple fixed-point classification
2. **CIFAR-10** — Validate visual convergence
3. **Simple language modeling** — Predict next token from equilibrium
4. **Reasoning tasks** — Leverage adaptive compute depth

---

## Pseudocode

### Training Loop

```python
def train_step(model, x, y, beta=0.1, max_iters=50, tol=1e-5):
    # === FREE PHASE ===
    h = initialize_hidden(x)
    for t in range(max_iters):
        h_new = model.forward(h, x)
        if norm(h_new - h) < tol:
            break
        h = h_new
    h_free = h.detach()
    
    # Store free-phase activations
    activations_free = get_layer_activations(model, h_free, x)
    
    # === NUDGED PHASE ===
    h = h_free.clone()
    for t in range(max_iters):
        h_new = model.forward(h, x)
        output = model.output_head(h_new)
        nudge = beta * (y - output)          # Nudge toward target
        h_new = h_new + project_nudge(nudge)  # Apply to relevant layers
        if norm(h_new - h) < tol:
            break
        h = h_new
    h_nudged = h
    
    # Store nudged-phase activations
    activations_nudged = get_layer_activations(model, h_nudged, x)
    
    # === WEIGHT UPDATE (Contrastive Hebbian) ===
    for param, act_free, act_nudged in zip(model.parameters(), 
                                            activations_free, 
                                            activations_nudged):
        # Local Hebbian update
        delta = (outer(act_nudged, act_nudged) - outer(act_free, act_free)) / beta
        param.grad = -delta  # Negative because we minimize loss
    
    optimizer.step()
```

### Inference

```python
def inference(model, x, max_iters=100, tol=1e-6):
    h = initialize_hidden(x)
    for t in range(max_iters):
        h_new = model.forward(h, x)
        if norm(h_new - h) < tol:
            print(f"Converged in {t+1} iterations")
            break
        h = h_new
    return model.output_head(h)
```

---

## Potential Challenges & Mitigations

### Convergence

| Challenge | Severity | Mitigation |
|-----------|----------|------------|
| Spectral norm ≥ 1 | 🔴 High | Spectral normalization, residual scaling |
| Oscillatory dynamics | 🟡 Medium | Damping, momentum-free updates |
| Slow convergence | 🟡 Medium | Anderson acceleration, learned initialization |

### β-Scaling

| Challenge | Severity | Mitigation |
|-----------|----------|------------|
| Too large β → biased gradients | 🟡 Medium | Start small (β=0.01), validate against BP |
| Too small β → noisy gradients | 🟡 Medium | Larger batch sizes, gradient accumulation |

**Recommended schedule**: Linear warmup from β=0.001 to β=0.1 over first 10% of training.

### Sequential Data

| Challenge | Severity | Mitigation |
|-----------|----------|------------|
| Temporal dependencies | 🟡 Medium | Inject input x persistently at each iteration |
| Causal structure | 🟢 Low | Use causal attention masking as usual |

---

## Open Research Questions

1. **Attention compatibility**: Self-attention is non-contractive. Does TorEqProp require specialized variants (linear attention, sparse attention, or attention with explicit regularization)?

2. **Energy formulation for attention**: Can we derive a rigorous energy function for the full transformer block including softmax attention?

3. **Scaling laws**: How does convergence iteration count scale with model size? Does this affect the O(1) memory advantage in practice?

4. **β-annealing**: What is the optimal schedule for β? Is there a principled way to adapt β during training?

5. **Comparison with modern Hopfield networks**: How does TorEqProp relate to the polynomial energy Hopfield formulations that underlie modern associative memory transformers?

6. **Non-equilibrium regimes**: Can useful learning occur before strict convergence? (cf. "early exit" strategies)

---

## Limitations

> [!CAUTION]
> These are known limitations that future research should address.

1. **Unvalidated**: No empirical results yet — this is purely theoretical.

2. **Convergence uncertainty**: Standard transformer blocks may not satisfy contraction conditions without significant architectural modifications.

3. **Computational overhead**: While memory is O(1), wall-clock time may exceed BP if many iterations are needed.

4. **Attention energy problem**: No known closed-form energy for softmax attention; this may require approximations or alternative attention mechanisms.

5. **Hyperparameter sensitivity**: β, damping factor, and convergence tolerance may require extensive tuning.

6. **Limited to classification-like tasks initially**: Extension to generative modeling (autoregressive LLMs) requires careful design of the nudging mechanism.

---

## Roadmap

### Phase 1: Proof of Concept
- [ ] Implement basic looped transformer with fixed-point iteration
- [ ] Validate convergence on MNIST
- [ ] Implement EqProp weight updates
- [ ] Compare gradients to BP (small β limit)

### Phase 2: Validation
- [ ] Scale to CIFAR-10
- [ ] Benchmark convergence speed vs. accuracy trade-off
- [ ] Experiment with different relaxation solvers
- [ ] Tune β-annealing schedules

### Phase 3: Language
- [ ] Adapt for next-token prediction
- [ ] Test on small language modeling benchmarks (WikiText-2)
- [ ] Explore adaptive compute for variable-difficulty inputs

### Phase 4: Scale
- [ ] Investigate scaling laws for TorEqProp
- [ ] Neuromorphic hardware simulation/deployment
- [ ] Compare to DEQ and Hopfield Transformer baselines

### Phase 5: Extensions
- [ ] Multi-modal inputs (vision-language)
- [ ] Reinforcement learning integration
- [ ] Online/continual learning scenarios

---

## Related Work

### Looped/Universal Transformers
- Dehghani et al. (2018) — Universal Transformers
- Recent 2024–2025 works on expressive power and reasoning with weight-tied transformers

### Equilibrium Propagation
- Scellier & Bengio (2017) — Original EqProp formulation
- Laborieux et al. (2021) — Scaling EqProp to modern architectures

### Deep Equilibrium Models (DEQ)
- Bai et al. (2019) — DEQ for sequence modeling
- Bai et al. (2020) — Multiscale DEQ

### Energy-Based Transformers
- Ramsauer et al. (2020) — Hopfield Networks is All You Need
- Hoover et al. (2023) — Energy Transformer

### Biologically Plausible Learning
- Lillicrap et al. (2020) — Backpropagation and the brain
- Whittington & Bogacz (2017) — Predictive coding approximates BP

---

## Citation

If you use this specification in your research, please cite:

```bibtex
@misc{toreqprop2024,
  title={Toroidal Equilibrium Propagation for Transformers: A Theoretical Specification},
  author={[Author]},
  year={2024},
  note={Theoretical research specification},
  url={[URL]}
}
```

---

<div align="center">

**TorEqProp** — Toward symmetric, local, biologically plausible transformer training.

</div>