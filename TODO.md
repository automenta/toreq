# TorEqProp Research TODO

> **Current Status**: EqProp validated at 92.7% MNIST (BP: 97.2%), gradient equiv 0.9972
> **Gap to Close**: 5% accuracy, O(1) memory advantage

---

## 🎯 Priority Matrix

| Option | Effort | Impact | Risk | Recommendation |
|--------|--------|--------|------|----------------|
| A. Fix Symmetric Mode | Medium | High | Medium | ⭐ Try first |
| B. Local Hebbian Update | High | High | Low | ⭐ Key differentiator |
| C. CIFAR-10 Scaling | Low | Medium | Low | Quick win |
| D. Multi-Layer Toroid | Medium | Medium | Medium | Architecture search |
| E. Anderson Acceleration | Low | Low | Low | Optimization |

---

## Option A: Fix Symmetric Mode (Restore Theoretical Foundation)

**Problem**: Tanh saturation (96.7%) kills gradients despite verified equivalence.

### Approaches

1. **Scaling the hidden state**
   ```python
   # Scale down before tanh to prevent saturation
   return torch.tanh(0.3 * (h + attn_out + ffn_out))
   ```

2. **Alternative bounded activations**
   - `hardtanh` (linear region)
   - `softsign`: x / (1 + |x|) — softer saturation
   - `gelu` with clipping

3. **Residual scaling**
   ```python
   # Damped residual to reduce energy
   return h + 0.1 * torch.tanh(attn_out + ffn_out)
   ```

4. **Better initialization**
   - Xavier/uniform with smaller scale
   - Zero-init residual branches

### Validation
- [ ] Saturation < 20%
- [ ] Training accuracy > 80%
- [ ] Gradient equiv maintained > 0.99

---

## Option B: Truly Local Hebbian Update (O(1) Memory)

**Goal**: Implement Algorithm 1b from README for true O(1) memory.

### Implementation in `src/updates.py`

```python
class LocalHebbianUpdate(UpdateStrategy):
    """No autodiff — direct weight updates from activation products."""
    
    def compute_model_update(self, model, h_free, h_nudged, x):
        with torch.no_grad():
            delta = (h_nudged - h_free) / self.beta
            # Accumulate Hebbian: ΔW = η * post * pre^T
            for name, param in model.named_parameters():
                if 'weight' in name:
                    # Need layer-specific pre/post activations
                    param.grad = self._hebbian_grad(name, delta, h_free)
        return None  # No loss tensor needed
```

### Challenges
- Need per-layer activation hooks
- Must match gradient direction (sign conventions)
- Validation: compare to MSE proxy gradients

### Validation
- [ ] Memory < 50% of BP
- [ ] Accuracy within 5% of MSE proxy

---

## Option C: CIFAR-10 Scaling (Quick Win)

**Goal**: Demonstrate scaling beyond MNIST.

### Changes Required
- [ ] Add `train_cifar.py` (copy from train_mnist.py)
- [ ] Patch input: 32×32×3 → patches → d_model
- [ ] May need d_model=256, more epochs

### Target
- BP baseline: ~65-70% (linear attention limited)
- EqProp target: within 5%

---

## Option D: Multi-Layer Toroid (Architecture Search)

**Goal**: Test L ∈ {2, 3, 4} block configurations.

### Implementation
```python
class MultiBlockToroid(nn.Module):
    def __init__(self, n_blocks, d_model, ...):
        self.blocks = nn.ModuleList([
            LoopedTransformerBlock(...) for _ in range(n_blocks)
        ])
    
    def forward(self, h, x):
        for block in self.blocks:
            h = block(h, x)
        return h
```

### Experiments
| L | Description | Expected |
|---|-------------|----------|
| 1 | Current | 92.7% |
| 2 | 2-block | +1-2%? |
| 3 | 3-block | +2-3%? slower convergence |
| 4 | 4-block | May not converge |

---

## Option E: Anderson Acceleration (Convergence Speed)

**Goal**: Reduce equilibrium iterations from 50 to 10-20.

### Implementation
```python
from scipy.optimize import anderson

def solve_anderson(self, f, h0, x, k=5):
    """Anderson acceleration with history k."""
    history = []
    h = h0
    for t in range(self.max_iters):
        h_new = f(h, x)
        history.append((h.clone(), h_new.clone()))
        if len(history) > k:
            history.pop(0)
        # Extrapolate from history
        h = self._extrapolate(history)
        if (h_new - h).norm() < self.tol:
            return h, t + 1
    return h, self.max_iters
```

---

## 📊 Decision Guide

### If goal is **publication-ready accuracy**:
1. Try Option A (fix symmetric) — validates theory
2. Try Option D (multi-layer) — improves capacity
3. Try Option C (CIFAR-10) — demonstrates scaling

### If goal is **differentiation from DEQ/BP**:
1. Option B (local Hebbian) — key claim
2. Memory profiling with truly local update

### If goal is **quick results**:
1. Option C (CIFAR-10) — 1-2 hours
2. Option E (Anderson) — faster iterations

---

## Quick Start Commands

```bash
# Current best
python train_mnist.py  # 92.7% in 5 epochs

# Verify gradient equiv still works
python test_gradient_equiv.py

# Memory comparison
python profile_memory.py

# Hyperparameter search
python hyperparam_sweep.py
```

---

## Success Criteria for Publication

| Criterion | Current | Target | Gap |
|-----------|---------|--------|-----|
| MNIST accuracy | 92.7% | ≥95% | 2.3% |
| Gradient equiv | 0.9972 | >0.99 | ✅ |
| Memory advantage | 1.04× (worse) | <0.5× | ❌ |
| CIFAR-10 | - | ≥65% | TODO |
