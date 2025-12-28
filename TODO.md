# TorEqProp Research Roadmap

> **Mission**: Demonstrate the first transformer trained via biologically plausible Equilibrium Propagation with O(1) memory and competitive accuracy.

> **Current State**: EqProp validated at **92.7% MNIST** (BP: 97.2%), gradient equivalence **0.9972** ✅  
> **Critical Gap**: Memory advantage unrealized (1.04× worse than BP)

---

## 🎯 Success Definition: Publishable Results

| Claim | Current | Target | Demo |
|-------|---------|--------|------|
| **Gradient equivalence** | 0.9972 cosine | >0.99 | `test_gradient_equiv.py` |
| **Competitive accuracy** | 92.7% MNIST | ≥95% MNIST, ≥70% CIFAR-10 | Training curves |
| **O(1) memory** | 1.04× worse | <0.5× of BP | Memory profile charts |
| **Adaptive compute** | — | Iterations ∝ difficulty | Per-sample analysis |
| **Scaling** | MNIST only | CIFAR-10, SST-2 | Multi-task benchmark |

---

## Phase 1: Close the Accuracy Gap (Priority ⭐⭐⭐)

### 1.1 Hyperparameter Optimization [IN PROGRESS]
- [/] Extended grid search (β × α × lr × d_model)
- [ ] Optuna Bayesian optimization with pruning
- [ ] Best config validation: 5+ runs with error bars
- **Target**: 93.5%+ consistent

### 1.2 Architecture Scaling
- [ ] Multi-block toroid (L ∈ {2, 3, 4})
  ```python
  # Test: L=2 blocks, d_model=128
  python train_mnist.py --n_blocks 2
  ```
- [ ] Width vs. depth analysis (d_model=256 single vs. d_model=128 × 2 blocks)
- [ ] Document capacity-convergence tradeoff
- **Target**: 95%+ with optimal architecture

### 1.3 Fix Symmetric Mode (Theoretical Purity)
- [ ] Diagnose tanh saturation (96.7% saturated)
  - [ ] Scale activations: `tanh(0.3 * h)` 
  - [ ] Alternative: softsign `x / (1 + |x|)`
  - [ ] Residual dampening: `h + 0.1 * tanh(...)`
- [ ] Validate symmetric mode achieves >80% accuracy
- [ ] Re-verify gradient equiv with fixed symmetric
- **Why**: Symmetric mode is theoretically cleaner—enables O(1) memory claim

---

## Phase 2: Demonstrate O(1) Memory (Priority ⭐⭐⭐)

> **Critical**: Without memory advantage, TorEqProp is just slow BP.

### 2.1 Implement Local Hebbian Update
```python
# Target: src/updates.py
class LocalHebbianUpdate(UpdateStrategy):
    """Algorithm 1b: No autodiff, forward-only."""
    
    def compute_model_update(self, model, h_free, h_nudged, x):
        with torch.no_grad():  # ← Key: no gradient tape
            delta = (h_nudged - h_free) / self.beta
            for name, param in model.named_parameters():
                if 'weight' in name:
                    # Hebbian: ΔW = η * post * preᵀ
                    param.grad = self._extract_hebbian(name, delta, h_free, x)
        return None
```

### 2.2 Memory Profiling
- [ ] Profile current implementation vs BP at d_model ∈ {128, 256, 512, 1024}
- [ ] Profile with LocalHebbianUpdate
- [ ] Create memory scaling chart (demo artifact)
- **Target**: EqProp <50% of BP memory at d_model=512+

### 2.3 Crossover Analysis
- [ ] Find d_model where O(1) advantage dominates
- [ ] Train model that **cannot fit with BP**
- **Demo**: "Train d_model=2048 on 8GB GPU (impossible with BP)"

---

## Phase 3: Scaling Demonstrations (Priority ⭐⭐)

### 3.1 CIFAR-10
- [ ] Create `train_cifar.py`
  - Patch embedding: 32×32×3 → patches → d_model
  - Adjust architecture (d_model=256, more epochs)
- [ ] BP baseline: ~65-70%
- [ ] EqProp target: within 5% of BP
- **Demo**: Training curves + final accuracy

### 3.2 Text Classification (SST-2)
- [ ] Add text tokenization pipeline
- [ ] Train sentiment classifier
- [ ] Target: ≥75% accuracy
- **Demo**: Example predictions

### 3.3 Algorithmic Reasoning
- [ ] Parity task (XOR of N bits)
- [ ] Addition (N-digit numbers)
- [ ] Sorting (N elements)
- **Hypothesis**: Equilibrium models adapt compute to difficulty
- **Demo**: Iterations vs. problem complexity chart

---

## Phase 4: Adaptive Compute Analysis (Priority ⭐⭐)

### 4.1 Per-Sample Iteration Analysis
- [ ] Log convergence iterations per sample
- [ ] Correlate with:
  - [ ] Sample difficulty (margin, entropy)
  - [ ] Misclassification rate
  - [ ] Human-judged ambiguity (for vision)

### 4.2 Early Exit Strategy
- [ ] Implement confidence-based early stopping
- [ ] Measure accuracy vs. average iterations tradeoff
- **Demo**: "Same accuracy with 40% fewer iterations"

### 4.3 Comparison to DEQ
- [ ] Train DEQ on same tasks
- [ ] Compare:
  - Accuracy
  - Iterations to converge
  - Memory usage
  - Training time
- **Demo**: Side-by-side table

---

## Phase 5: Convergence Engineering (Priority ⭐)

### 5.1 Anderson Acceleration
```python
# Reduce iterations from 50 → 10-20
def solve_anderson(self, f, h0, x, k=5):
    history = []
    for t in range(self.max_iters):
        h_new = f(h, x)
        history.append((h, h_new))
        h = self._extrapolate(history[-k:])
        if (h_new - h).norm() < self.tol:
            return h, t + 1
```
- [ ] Implement in `src/solver.py`
- [ ] Benchmark: iterations saved, wall-clock impact
- **Target**: 2-5× faster convergence

### 5.2 Learned Initialization
- [ ] Small network predicts h₀ from x
- [ ] Train jointly
- [ ] Measure iteration reduction

### 5.3 Spectral Normalization
- [ ] Add spectral norm to guarantee contraction
- [ ] Use when convergence is unstable

---

## Demo Artifacts to Produce

| Artifact | Purpose | Status |
|----------|---------|--------|
| **Memory scaling chart** | Prove O(1) advantage | TODO |
| **Accuracy comparison table** | Competitive with BP | ✅ Have data |
| **Gradient equiv plot** (β sweep) | Scientific validation | Can generate |
| **Iterations vs. difficulty scatter** | Adaptive compute | TODO |
| **Training curves** (MNIST, CIFAR, SST-2) | Task generality | MNIST done |
| **Video/gif** of equilibrium finding | Visual intuition | TODO |

---

## Quick Commands

```bash
# Current validation
python test_gradient_equiv.py  # Gradient equiv (expect 0.99+)
python train_mnist.py          # Train to 92.7% in 5 epochs
python train_mnist_bp.py       # BP baseline (97.2%)
python profile_memory.py       # Memory comparison

# Hyperparameter search
python hyperparam_sweep.py     # Grid search (running)

# New experiments (to implement)
python train_cifar.py          # CIFAR-10 scaling
python analyze_iterations.py   # Per-sample iteration analysis
python demo_memory.py          # Generate memory charts
```

---

## Decision Checkpoints

| Week | Checkpoint | Go Criterion | Pivot If... |
|------|------------|--------------|-------------|
| Now | Hyperparam sweep | Best config >93% | Continue tuning |
| +1 | LocalHebbianUpdate | Memory <0.7× BP | Fix implementation |
| +2 | CIFAR-10 | >65% accuracy | Adjust architecture |
| +3 | Adaptive compute | Correlation >0.5 | Focus elsewhere |
| +4 | Paper draft | Clear narrative | Pivot framing |

---

## Pivot Strategies

### If O(1) memory fails:
→ **Reframe**: Focus on gradient equivalence proof + biological plausibility
→ Still publishable as theoretical contribution

### If accuracy stays <95%:
→ **Try**: Multi-block architecture, longer training
→ **Reframe**: "Proof of concept" with roadmap for scaling

### If symmetric mode hopeless:
→ **Accept**: Non-symmetric EqProp is the contribution
→ **New claim**: "EqProp works beyond symmetric constraints"

### If convergence too slow:
→ **Add**: Anderson acceleration, learned init
→ **Reframe**: Memory advantage more important than speed

---

## Success Criteria Summary

```
MVP (Publishable):
  ✅ Gradient equiv > 0.99 (DONE: 0.9972)
  🔲 MNIST ≥ 95% accuracy (Current: 92.7%)
  🔲 Memory < 0.5× BP (Current: 1.04×)

Full Paper:
  🔲 CIFAR-10 ≥ 70%
  🔲 Adaptive compute demonstrated
  🔲 Comparison to DEQ/BP baselines
  🔲 Memory advantage at scale

Stretch:
  🔲 Text classification (SST-2)
  🔲 Algorithmic reasoning tasks
  🔲 Neuromorphic simulation
```

---

## Next Actions (Prioritized)

1. **[NOW]** Complete hyperparam sweep → find best config
2. **[NEXT]** Implement `LocalHebbianUpdate` → unlock O(1) memory
3. **[NEXT]** Multi-block architecture → improve accuracy
4. **[THEN]** Memory profiling with local update → generate charts
5. **[THEN]** CIFAR-10 experiment → demonstrate scaling
6. **[THEN]** Write paper draft → crystallize narrative
