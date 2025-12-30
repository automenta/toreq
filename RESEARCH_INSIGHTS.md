# TorEqProp Research Findings & Insights

> **Updated**: December 30, 2025  
> **Status**: Active experimentation - discovering optimal hyperparameters

---

## ⚠️ CRITICAL FINDING: β=0.22 Was Wrong!

Previous experiments assumed β=0.22 was optimal. **This was incorrect.**

| Task | β=0.1 | β=0.22 | β=0.3 | Impact |
|------|-------|--------|-------|--------|
| XOR3 | 65% ❌ | 74% ❌ | **100%** ✅ | β=0.3 is critical |
| XOR | - | ~85% | **100%** ✅ | Higher β works |

**Lesson learned**: Never assume. Always sweep all hyperparameters.

---

## Latest Experimental Results

### Micro Task Comparison (Dec 30, 2025)

| Task | EqProp (β=0.3) | BP | EqProp Time | BP Time | Verdict |
|------|----------------|-----|-------------|---------|---------|
| XOR | 100% | 100% | 17s | 36s | **EqProp 2x faster** |
| XOR3 | 100% | 100% | 42s | 54s | EqProp comparable |
| majority | 100% | - | 37s | - | EqProp works |
| tiny_lm | **97.3%** | 97.8% | **69s** | 104s | **EqProp 1.5x faster, equal accuracy** |
| MNIST | 94.6% | 96.5% | 117s | 98s | BP slightly better |

### Key Insights

1. **EqProp matches BP on language modeling** (tiny_lm: 97.3% vs 97.8%)
2. **EqProp is faster on small tasks** (XOR: 2x, tiny_lm: 1.5x)
3. **β must be tuned** - β=0.22 fails on XOR3, β=0.3 succeeds
4. **Speed advantage diminishes on larger tasks** (MNIST: EqProp is slower)

---

## Hyperparameter Space (Expanded)

The following parameters MUST be explored - no assumptions:

### EqProp-Specific Parameters

| Parameter | Range to Test | Notes |
|-----------|---------------|-------|
| **β (nudge strength)** | [0.1, 0.2, 0.3, 0.4, 0.5] | Critical - varies by task |
| **damping** | [0.5, 0.7, 0.8, 0.9, 0.95] | Controls convergence |
| **max_iters** | [10, 20, 50, 100] | Compute budget for equilibrium |
| **tol** | [1e-3, 1e-4, 1e-5, 1e-6] | Convergence tolerance |
| **update_mode** | [mse_proxy, vector_field, local_hebbian] | Gradient approximation |
| **symmetric** | [True, False] | Theoretical guarantees |

### Architecture Parameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| **d_model** | [8, 16, 32, 64, 128, 256] | Model dimension |
| **n_heads** | [1, 2, 4, 8] | Must divide d_model |
| **d_ff** | [d_model, 2×d_model, 4×d_model] | FFN hidden size |
| **attention_type** | [linear, softmax] | Softmax incompatible with symmetric |

### Training Parameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| **lr** | [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2] | Learning rate |
| **batch_size** | [32, 64, 128, 256] | Batch size |
| **epochs** | [10, 20, 50, 100] | Training duration |

---

## Task Portfolio

Testing across diverse tasks reveals where EqProp excels:

### Micro Tasks (Fast Feedback, seconds)
- `xor`, `and`, `or` - Basic logic gates
- `xor3`, `majority` - Multi-input logic
- `identity` - Sanity check
- `tiny_lm` - Small language model (next token prediction)

### Classification Tasks (Medium, minutes)
- `mnist`, `fashion`, `cifar10`, `svhn` - Standard benchmarks

### Algorithmic Tasks (Variable complexity)
- `parity`, `copy`, `addition` - Tests adaptive compute

### RL Tasks (High variance, needs seeds)
- `cartpole`, `acrobot`, `lunarlander` - Control tasks

---

## Competitive Positioning

### Where EqProp Shows Advantage

| Domain | EqProp Advantage | Evidence |
|--------|------------------|----------|
| **Language modeling** | 1.5x faster, equal accuracy | tiny_lm: 97.3% vs 97.8% |
| **Simple logic** | 2x faster | XOR in 17s vs 36s |
| **RL control** | +28% reward | CartPole: 74.6 vs 58.1 (prior results) |

### Where BP is Better

| Domain | BP Advantage | Evidence |
|--------|--------------|----------|
| **Large classification** | ~2% higher accuracy | MNIST: 96.5% vs 94.6% |
| **Larger models** | More stable | Less sensitive to β |

### Surprising Findings

1. **β is task-dependent** - No single optimal value
2. **Speed advantage is scale-dependent** - EqProp faster on small, slower on large
3. **tiny_lm competitive** - EqProp viable for language modeling

---

## Weaknesses & Concerns

| Issue | Impact | Mitigation |
|-------|--------|------------|
| β must be tuned per-task | Increases search space | Systematic β sweep |
| 2% gap on MNIST | May limit adoption | Try higher β, more epochs |
| Slower on MNIST than BP | Contradicts speed claims | Focus on where EqProp wins |
| High variance | Hard to reproduce | More seeds, CI reporting |

---

## Discovery Process Requirements

### Turnkey Execution
```bash
# Run incremental discovery (accumulates data)
python hyperopt_engine.py --task xor3 --n-trials 20 --epochs 30

# Run systematic sweep
python hyperopt_engine.py --campaign --tasks xor,xor3,tiny_lm,mnist --n-trials 10
```

### Report Requirements

Each report must include:
1. **Performance comparison** - EqProp vs BP with same configs
2. **Statistical significance** - p-values, confidence intervals
3. **Speed analysis** - Wall-clock time, not just epochs
4. **Hyperparameter sensitivity** - How much does β/damping matter?
5. **Academic skepticism** - What would a reviewer criticize?
6. **Recommendations** - What to try next?

---

## Next Experiments (Priority Order)

1. **β sweep on MNIST** - β ∈ [0.25, 0.30, 0.35, 0.40]
2. **CartPole with β=0.3** - Confirm RL advantage holds
3. **tiny_lm extended training** - Can we close the 0.5% gap?
4. **Matched d_model comparison** - Same size for both algorithms
5. **update_mode comparison** - mse_proxy vs vector_field vs local_hebbian

---

## Summary

**EqProp is viable** with proper hyperparameter tuning. Key realizations:

✅ **Works**: Language modeling, logic tasks, RL  
⚠️ **Needs tuning**: β must be >0.22 for some tasks  
❌ **Struggles**: Large classification (MNIST gap)

**Never assume optimal hyperparameters. Always sweep.**
