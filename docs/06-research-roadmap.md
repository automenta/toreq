# Research Roadmap

> **Mission**: Train transformers via biologically plausible Equilibrium Propagation, demonstrating gradient equivalence, competitive accuracy, O(1) memory potential, and adaptive compute — any of which alone is publishable.

---

## Current Status

| Claim | Current | Target | Publishable Alone? |
|-------|---------|--------|-------------------|
| Gradient equivalence | **0.9972** ✅ | >0.99 | ✅ Yes (theoretical validation) |
| MNIST accuracy | **92.11%** ✅ | ≥95% | ✅ Yes (first EqProp transformer) |
| O(1) memory | **Activated** ✅ | <0.5× BP | ✅ Yes (hardware implications) |
| Adaptive compute | **Tooling ready** 🔄 | Demonstrated | ✅ Yes (novel dynamics) |
| Biological plausibility | ✅ Validated | Documented | ✅ Yes (neuroscience connection) |
| **β=0.25 optimal** | **Discovered** 🆕 | Documented | ✅ Yes (counterintuitive finding) |

---

## Research Tracks (Parallel)

### Track A: Accuracy to 95%+ (Days 1-2)

| Step | Action | Time | Expected | Status |
|------|--------|------|----------|--------|
| A1 | d_model=256 with best config | 2 hr | +1-2% | ✅ Done |
| A2 | Add dropout=0.1 | 30 min | +0.5% | ✅ Done |
| A3 | β annealing (0.3→**0.25**) ⚠️ CORRECTED | 1 hr | +0.5% | 🔄 Rerun needed |
| A4 | Multi-seed validation | 4 hr | Mean ≥95% | ⏳ Ready |
| A5 | Document best config | 30 min | Paper table | ⏳ Pending |

**CRITICAL FINDING**: β=0.2 causes training collapse. Keep β≥0.23 for stability.

```bash
# CORRECTED: End at β=0.25 instead of 0.2
python train.py --d-model 256 --n-heads 8 --d-ff 1024 \
    --beta 0.25 --damping 0.8 --lr 0.002 --epochs 12 \
    --dropout 0.1 --beta-anneal --compile
```

### Track B: O(1) Memory Demo (Days 2-3)

| Step | Action | Time | Expected | Status |
|------|--------|------|----------|--------|
| B1 | Activate full LocalHebbianUpdate | 2 hr | Bypass autodiff | ✅ Done |
| B2 | Profile d_model={256,512,1024,2048} | 1 hr | <0.5× BP ratio | ⏳ Ready |
| B3 | "Impossible demo" | 1 hr | d_model=2048 trains | ⏳ Ready |
| B4 | Memory scaling plot | 30 min | Paper figure | ⏳ Ready |

**Status**: Pure Hebbian updates activated. No autodiff for model parameters.

### Track C: Adaptive Compute Analysis (Days 3-4)

| Step | Action | Time | Expected |
|------|--------|------|----------|
| C1 | Log per-sample iterations | 1 hr | Data collection |
| C2 | Correlate iters vs margin | 1 hr | Strong correlation |
| C3 | Iterations per digit class | 1 hr | 4,9 harder |
| C4 | Early exit analysis | 2 hr | 30-50% compute savings |
| C5 | Visualize h_t trajectory | 2 hr | "Thinking" dynamics |

### Track D: Scaling (Days 4-5)

| Step | Action | Time | Expected |
|------|--------|------|----------|
| D1 | CIFAR-10 patch embedding | 2 hr | Implementation |
| D2 | CIFAR-10 training | 4 hr | >65% accuracy |
| D3 | SST-2 text (optional) | 4 hr | >77% accuracy |
| D4 | Algorithmic reasoning | 3 hr | Parity/addition |

### Track E: Algorithmic & Structured Tasks (Days 5-6)

| Task | Description | Why EqProp Excels |
|------|-------------|-------------------|
| **Parity** | XOR of N bits | Requires N sequential ops |
| **Addition** | Add N-digit numbers | Carry propagation iterative |
| **Copying** | Repeat sequence | Tests equilibrium memory |
| **Sorting** | Sort N numbers | Comparison chains |

**Hypothesis**: Equilibrium iteration count correlates with problem structure.

---

## Timeline

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | Accuracy push | d_model=256 → 95%+ |
| 2 | O(1) memory | LocalHebbianUpdate demo |
| 3 | Adaptive compute | Per-sample iteration analysis |
| 4 | Multi-seed + CIFAR | Validation + scaling |
| 5 | Analysis | All figures generated |
| 6 | Paper draft | Submission-ready manuscript |

### Extended Timeline (8 weeks)

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

## Checkpoint Decision Points

| Week | Checkpoint | Go/No-Go Criterion | Pivot If... |
|------|------------|---------------------|-------------|
| 2 | Gradient check | Cosine >0.95 with softmax OR >0.99 with linear | Softmax fails → Pivot A |
| 3 | MNIST baseline | >90% accuracy, <100× slowdown | Accuracy low → Pivot D |
| 4 | MNIST complete | >95% accuracy OR clear pivot narrative | Slowdown high → Pivot B |
| 6 | Scaling | CIFAR >65% OR compelling memory analysis | Neither → focus on theory |
| 8 | Final | Clear publication narrative identified | Always: write the paper |

---

## Success Definition

**Minimum Publishable Result** (any ONE of):
- 95%+ accuracy with gradient equivalence
- O(1) memory demonstrated at scale
- Adaptive compute correlation proven
- Novel β>0 insight explained

**Maximum Result** (all of):
- 95%+ MNIST, 65%+ CIFAR-10
- O(1) memory with d_model=2048
- Adaptive compute quantified
- Theoretical analysis included
- DEQ comparison table

---

## Fallback Strategies

| If... | Then... | Still Publishable? |
|-------|---------|-------------------|
| 95% unreachable | Focus 94% + adaptive compute | ✅ Yes |
| O(1) memory hard | Position as "towards O(1)" | ✅ Yes |
| CIFAR-10 fails | MNIST + algorithmic tasks | ✅ Yes |
| Adaptive compute weak | Emphasize bio-plausibility | ✅ Yes |
| All fails | Negative result paper | ✅ Yes (rare) |
