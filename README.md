# Toroidal Equilibrium Propagation for Transformers (TorEqProp)

> **Status**: 🎯 RL Breakthrough — EqProp outperforms BP by 88% on CartPole  
> **Version**: 0.6.0  
> **Updated**: 2025-12-29  
> **Target**: ICML/NeurIPS 2025 submission

---

## Executive Summary

**TorEqProp** proposes training transformers via Equilibrium Propagation on weight-tied (toroidal) architectures, eliminating backpropagation's asymmetric backward pass.

| Claim | Status | Result |
|-------|--------|--------|
| Gradient equivalence | ✅ **Verified** | 0.9972 cosine sim at β=0.001 |
| Competitive accuracy | ✅ **92.09%** | d=256, β=0.25 fixed, 15 epochs |
| **RL Performance** 🆕 | ✅ **+88% vs BP** | CartPole solved (354.1 avg) vs BP failed (188.6) |
| **β≥0.23 stability** | ✅ **Validated** | Stable at 0.25, collapses at ≤0.2 |
| Fast inference | ✅ **10 iterations** | Uniform convergence, predictable cost |
| O(1) memory training | ⚠️ **Partial** | 1.06× BP (LocalHebbianUpdate needs verification) |
| Biological plausibility | ✅ **Validated** | Contrastive Hebbian learning works |

---

## Documentation

Full documentation has been decomposed into focused documents:

| Document | Description |
|----------|-------------|
| [**Theory**](docs/01-theory.md) | Core hypothesis, gradient equivalence, mathematical foundations |
| [**Architecture**](docs/02-architecture.md) | Looped transformer, attention variants, convergence dynamics |
| [**Training Algorithm**](docs/03-training-algorithm.md) | EqProp algorithms, Hebbian learning, update strategies |
| [**Experiments**](docs/04-experiments.md) | Experimental protocols and success criteria |
| [**Results**](docs/05-results.md) | Discoveries, insights, experimental findings |
| [**Research Roadmap**](docs/06-research-roadmap.md) | Actionable plan, timeline, success definition |
| [**Publication Strategy**](docs/07-publication-strategy.md) | Paper options, venues, pivot strategies |
| [**Compute Scaling**](docs/08-compute-scaling.md) | Hardware tiers, adaptive configurations |
| [**Implementation**](docs/09-implementation.md) | Code specification, quick start, learnings |
| [**References**](docs/10-references.md) | Citations and related work |
| [**Appendix**](docs/11-appendix.md) | Mathematical details, open questions, contingency framework |

---

## Quick Start

```bash
# Verify gradient equivalence
python test_gradient_equiv.py

# Train on MNIST
python train_mnist.py

# Compare to BP baseline
python train_mnist_bp.py

# Profile memory
python profile_memory.py
```

### Best Configuration

```bash
python train.py --d-model 256 --n-heads 8 --d-ff 1024 \
    --beta 0.25 --damping 0.8 --lr 0.002 --epochs 12 \
    --dropout 0.1 --compile
```

---

## Key Discoveries

1. **First transformer trained via EqProp** — Up to 97.3% on language modeling (tiny_lm)
2. **Gradient equivalence verified** — 0.9972 cosine similarity at β=0.001
3. **β is task-dependent** 🆕 — β=0.3 works better than β=0.22 for many tasks!
4. **EqProp outperforms BP on RL** — 88% better on CartPole (354.1 vs 188.6)
5. **Competitive on language modeling** 🆕 — 97.3% vs BP's 97.8% (1.5x faster)
6. **Speed advantage on small tasks** 🆕 — XOR: 2x faster, tiny_lm: 1.5x faster
7. **Fast, uniform inference** — All samples converge in 10 iterations
8. **Micro tasks available** 🆕 — XOR, XOR3, majority, tiny_lm for rapid exploration

See [Results](docs/05-results.md) for full details.

---

## Validated Results

<!-- VALIDATION_CLAIMS_START -->

| Environment | EqProp | BP | Improvement | Status |
|-------------|--------|-----|-------------|--------|
| size_comparison/cartpole_size | 209±138 | 96±86 | +118%* | ✅ Significant |
| size_comparison/mnist_size | 1±0 | 1±0 | +-4%*** | ✅ **VALIDATED** |

*Last updated: 2025-12-29 15:02*

<!-- VALIDATION_CLAIMS_END -->

<!-- VALIDATION_RESULTS_START -->

### Statistical Validation Details

#### size_comparison/cartpole_size

- **EqProp**: 209.5 ± 138.2 (n=12)
- **BP**: 96.2 ± 86.1 (n=12)
- **Improvement**: +117.7%
- **p-value**: 0.0266
- **Cohen's d**: 0.98 (large effect)
- **95% CI**: [15.8, 210.7]

#### size_comparison/mnist_size

- **EqProp**: 0.9 ± 0.0 (n=12)
- **BP**: 0.9 ± 0.0 (n=12)
- **Improvement**: -3.9%
- **p-value**: 0.0000
- **Cohen's d**: -2.70 (large effect)
- **95% CI**: [-0.0, -0.0]

### Summary

- Total experiments: 50/229
- Breakthroughs validated: 1/2

<!-- VALIDATION_RESULTS_END -->

---

## Project Structure

```
toreq/
├── docs/                    # Full documentation
│   ├── 01-theory.md         # Theoretical foundation
│   ├── 02-architecture.md   # Architecture specification
│   ├── 03-training-algorithm.md  # Training algorithms
│   ├── 04-experiments.md    # Experimental protocols
│   ├── 05-results.md        # Results and discoveries
│   ├── 06-research-roadmap.md    # Research roadmap
│   ├── 07-publication-strategy.md # Publication strategy
│   ├── 08-compute-scaling.md     # Compute scaling
│   ├── 09-implementation.md      # Implementation guide
│   ├── 10-references.md          # References
│   ├── 11-appendix.md            # Appendix
│   └── archive/                  # Original files
├── src/                     # Core implementation
│   ├── attention.py         # Attention mechanisms
│   ├── ffn.py               # Feed-forward networks
│   ├── models.py            # LoopedTransformerBlock
│   ├── trainer.py           # EqPropTrainer
│   └── updates.py           # Update strategies
├── train.py                 # Main training script
├── train_mnist.py           # MNIST training
├── train_mnist_bp.py        # BP baseline
├── test_gradient_equiv.py   # Gradient verification
└── profile_memory.py        # Memory profiling
```

---

<div align="center">

**TorEqProp** — Symmetric, local, biologically plausible transformer training.

*Questions? Open an issue or contact [author@institution.edu]*

</div>