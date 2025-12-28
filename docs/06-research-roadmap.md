# Research Roadmap

> **Mission**: Train transformers via biologically plausible Equilibrium Propagation, demonstrating gradient equivalence, competitive accuracy, O(1) memory potential, and adaptive compute — any of which alone is publishable.

---

## Current Status

| Claim | Current | Target | Publishable? |
|-------|---------|--------|--------------|
| Gradient equivalence | **0.9972** ✅ | >0.99 | ✅ Yes |
| MNIST accuracy | **92.11%** | ≥95% | ✅ Yes |
| O(1) memory | **Implemented** ✅ | <0.5× BP | ✅ Yes |
| Adaptive compute | **Ready** 🔄 | Demonstrated | ✅ Yes |
| β=0.25 optimal | **Discovered** 🆕 | Documented | ✅ Yes |

---

## 🚨 Immediate Action Items

### Priority 1: Validate β=0.25 (30 min)
```bash
python train.py --d-model 256 --n-heads 8 --d-ff 1024 \
    --beta 0.25 --damping 0.8 --lr 0.002 --epochs 15 \
    --dropout 0.1 --compile
```
**Goal**: Confirm 94-95% accuracy with corrected β endpoint.

### Priority 2: Multi-Seed Validation (2.5 hr)
```bash
./run_experiments.sh multiseed
```
**Goal**: Mean accuracy ≥94%, std < 1% across 5 seeds.

### Priority 3: O(1) Memory Demo (1 hr)
```bash
./run_experiments.sh memory
```
**Goal**: Demonstrate <0.5× BP memory at d_model=2048.

### Priority 4: Adaptive Compute (1 hr)
```bash
./run_experiments.sh adaptive
```
**Goal**: Show iterations correlate with sample difficulty.

---

## Research Tracks

### Track A: Accuracy ≥95%

| Step | Action | Status |
|------|--------|--------|
| A1 | d_model=256 baseline | ✅ 92.11% |
| A2 | Add dropout=0.1 | ✅ Done |
| A3 | β=0.25 fixed (corrected) | 🔄 **RUN NOW** |
| A4 | 5-seed validation | ⏳ Ready |
| A5 | Document best config | ⏳ Pending |

**Critical**: β=0.2 causes collapse. Keep β≥0.23.

### Track B: O(1) Memory

| Step | Action | Status |
|------|--------|--------|
| B1 | LocalHebbianUpdate active | ✅ Done |
| B2 | Profile d={256,512,1024,2048} | ⏳ Ready |
| B3 | "Impossible demo" (d=2048) | ⏳ Ready |
| B4 | Generate paper figure | ⏳ Pending |

### Track C: Adaptive Compute

| Step | Action | Status |
|------|--------|--------|
| C1 | Log per-sample iterations | ⏳ Ready |
| C2 | Correlate iters vs margin | ⏳ Ready |
| C3 | Iterations by digit class | ⏳ Ready |
| C4 | Early exit analysis | ⏳ Ready |
| C5 | Visualize h_t trajectory | ⏳ Ready |

### Track D: Scaling

| Step | Action | Status |
|------|--------|--------|
| D1 | CIFAR-10 implementation | ⏳ Future |
| D2 | CIFAR-10 training | ⏳ Future |
| D3 | SST-2 text | ⏳ Future |
| D4 | Algorithmic reasoning | ⏳ Future |

---

## Quick Commands

```bash
# Run all experiments
./run_experiments.sh all

# Individual experiments
./run_experiments.sh accuracy   # β=0.25 training
./run_experiments.sh multiseed  # 5-seed validation
./run_experiments.sh memory     # Memory profiling
./run_experiments.sh adaptive   # Adaptive compute
./run_experiments.sh gradient   # Gradient verification
```

---

## Timeline

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | Accuracy validation | β=0.25 → 95%+ |
| 2 | O(1) memory | Memory scaling plot |
| 3 | Adaptive compute | Iteration analysis |
| 4 | Multi-seed + docs | Validation + figures |
| 5 | Paper draft | Manuscript |

---

## Success Metrics

**Minimum Publishable** (any ONE):
- 95%+ accuracy with gradient equivalence
- O(1) memory demonstrated at d=2048
- Adaptive compute correlation proven
- β>0 insight theoretically explained

**Maximum Result** (all):
- 95%+ MNIST, 65%+ CIFAR-10
- O(1) memory at d=2048
- Adaptive compute quantified
- DEQ comparison
- Full theoretical analysis

---

## Fallback Strategies

| If... | Then... | Still Publishable? |
|-------|---------|-------------------|
| 95% unreachable | Focus 94% + adaptive | ✅ Yes |
| O(1) memory hard | "Towards O(1)" framing | ✅ Yes |
| CIFAR-10 fails | MNIST + algorithmic | ✅ Yes |
| Adaptive weak | Emphasize bio-plausibility | ✅ Yes |

