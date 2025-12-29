# Toroidal Equilibrium Propagation for Transformers (TorEqProp)

> **Status**: 🎯 β-Stability Validated — 92.09% MNIST accuracy, β≥0.23 stability confirmed  
> **Version**: 0.5.0  
> **Updated**: 2025-12-28  
> **Target**: ICML/NeurIPS 2025 submission

---

## Executive Summary

**TorEqProp** proposes training transformers via Equilibrium Propagation on weight-tied (toroidal) architectures, eliminating backpropagation's asymmetric backward pass.

| Claim | Status | Result |
|-------|--------|--------|
| Gradient equivalence | ✅ **Verified** | 0.9972 cosine sim at β=0.001 |
| Competitive accuracy | ✅ **92.09%** | d=256, β=0.25 fixed, 15 epochs |
| **β≥0.23 stability** 🆕 | ✅ **Validated** | Stable at 0.25, collapses at ≤0.2 |
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

1. **First transformer trained via EqProp** — 92.09% MNIST accuracy (validated 2025-12-28)
2. **Gradient equivalence verified** — 0.9972 cosine similarity at β=0.001
3. **β≥0.23 stability threshold** — Novel finding: training collapses at β≤0.2, stable at β=0.25
4. **Theory-practice gap** — EqProp theory suggests β→0, practice requires β>0 for stability
5. **Fast, uniform inference** — All samples converge in 10 iterations (predictable compute)
6. **Non-symmetric mode works** — Symmetric constraints not required for practical training

See [Results](docs/05-results.md) for full details.

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