# TorEqProp Research Roadmap

> **Mission**: First transformer trained via biologically plausible EqProp with O(1) memory.

| Claim | Current | Target | Blocker |
|-------|---------|--------|---------|
| Gradient equiv | **0.9972** ✅ | >0.99 | — |
| Accuracy | 92.7% | ≥95% MNIST | Hyperparam/arch |
| O(1) memory | 1.04× worse ❌ | <0.5× BP | LocalHebbianUpdate |
| Scaling | MNIST | +CIFAR-10 | train_cifar.py |

---

## 🚀 Efficiency Multipliers (Do First)

These unlock multiple downstream wins:

| Action | Effort | Unlocks |
|--------|--------|---------|
| **torch.compile** | 5 min | 20-40% faster experiments |
| **Wandb integration** | 30 min | Auto-logging, sweep viz, reproducibility |
| **Config dataclass** | 30 min | Clean CLI, easy sweeps, no magic numbers |
| **Unified train.py** | 1 hr | One script for MNIST/CIFAR/SST-2 via `--dataset` |

```python
# Config pattern to adopt
@dataclass
class Config:
    dataset: str = "mnist"
    d_model: int = 128
    beta: float = 0.1
    # ... auto-generates argparse
```

---

## Phase 1: Accuracy + Memory (Parallel Tracks)

### Track A: Close Accuracy Gap
| Step | Action | Time | Target |
|------|--------|------|--------|
| A1 | Finish hyperparam sweep | Running | Best β/α/lr |
| A2 | Test d_model=256 | 2 hr | +1-2% |
| A3 | Add 2nd block (L=2) | 2 hr | +1-2% |
| A4 | Combine best → validate 5 seeds | 4 hr | 95%+ mean |

### Track B: O(1) Memory (THE differentiator)
| Step | Action | Time | Target |
|------|--------|------|--------|
| B1 | Implement `LocalHebbianUpdate` | 4 hr | No autodiff |
| B2 | Profile at d_model={256,512,1024} | 1 hr | <0.5× BP |
| B3 | "Impossible with BP" demo | 1 hr | d_model=2048 trains |

**Key insight**: B1-B3 are more impactful than A2-A4. Memory claim is unique; accuracy can be improved later.

---

## Phase 2: Scaling (One Script, Three Datasets)

```bash
# Target: unified interface
python train.py --dataset mnist    # baseline
python train.py --dataset cifar10  # scaling
python train.py --dataset sst2     # text
```

| Dataset | New Code | Baseline | EqProp Target |
|---------|----------|----------|---------------|
| MNIST | — | 97.2% | 95%+ |
| CIFAR-10 | Patch embed | ~68% | 63%+ |
| SST-2 | Tokenizer | ~82% | 77%+ |

**Leverage**: Same training loop, just swap data loader + embedding.

---

## Phase 3: Analysis Artifacts (Paper Figures)

| Figure | Script | Data Needed | Est. Time |
|--------|--------|-------------|-----------|
| Memory scaling curve | `plot_memory.py` | B2 profile data | 30 min |
| β→0 gradient equiv | `plot_beta_sweep.py` | Already have | 30 min |
| Iterations vs difficulty | `analyze_iters.py` | Log per-sample | 2 hr |
| Training curves (3 datasets) | Wandb export | Phase 2 runs | Auto |

**Leverage**: Most figures auto-generate from existing/planned runs.

---

## Phase 4: Convergence Speed (If Needed)

Only pursue if iterations >30 average:

| Technique | ROI | Implement If... |
|-----------|-----|-----------------|
| Anderson acceleration | High | Avg iters >30 |
| Learned init | Medium | Marginal improvement from Anderson |
| Spectral norm | Low | Convergence unstable |

---

## 🎯 Revised Priority Stack

```
MUST (blocks publication):
  1. LocalHebbianUpdate → O(1) memory claim     [4 hr]
  2. Memory profile chart                        [1 hr]
  3. Best config → 95% MNIST                     [4 hr]

SHOULD (strengthens paper):
  4. CIFAR-10 scaling                            [3 hr]
  5. Iterations vs difficulty analysis           [2 hr]
  6. DEQ comparison table                        [3 hr]

COULD (stretch):
  7. SST-2 text classification                   [4 hr]
  8. Algorithmic reasoning (parity/addition)     [4 hr]
  9. Neuromorphic simulation                     [?]
```

---

## Quick Wins Queue

Things that take <1 hour but improve everything:

- [ ] Add `--compile` flag → 20% faster
- [ ] Log iterations per sample → enables Phase 3 analysis
- [ ] Export best config to `configs/best.yaml`
- [ ] Add `--seed` flag with multiple seeds for error bars
- [ ] Gradient clipping experiment (quick accuracy boost?)
- [ ] β scheduler: start 0.2 → anneal to 0.05
- [ ] Label smoothing (regularization)

---

## 🔀 Parallelization Strategy

Run simultaneously on different GPUs/sessions:

| Session | Task | Duration |
|---------|------|----------|
| GPU 0 | Hyperparam sweep (running) | ~3 hr |
| GPU 1 | LocalHebbianUpdate dev | 4 hr |
| CPU | Write train_cifar.py skeleton | 1 hr |
| CPU | Setup wandb + config dataclass | 1 hr |

---

## Decision Points

| Day | Check | Go | Pivot |
|-----|-------|----|----- |
| +1 | Sweep done | Best >93.5% | More tuning |
| +2 | LocalHebbian works | Memory <0.7× | Debug |
| +3 | CIFAR-10 baseline | >60% | Adjust arch |
| +5 | Paper outline | Clear story | Reframe claims |

---

## Pivot Playbook

| Failure | Response |
|---------|----------|
| Memory still 1×+ | Check: are we storing h history? Use checkpointing? |
| Accuracy stuck 92-93% | Try: dropout, LN placement, MLP ratio |
| CIFAR-10 fails | Simplify: focus MNIST + memory story |
| Convergence slow | Add: Anderson accel (2-day detour) |

---

## Future (Post-Publication)

### High-Yield Extensions
- Spiking EqProp → neuromorphic hardware
- Equilibrium VAE → generative modeling
- RL with equilibrium value function
- Theoretical: convergence bounds, expressiveness

### Quick Experiments (1-2 days)
- β annealing schedule
- Momentum in equilibrium (Polyak)
- Multi-scale loss at intermediate h
- Noise injection for robustness

---

## Commands

```bash
# Validation
python test_gradient_equiv.py
python train_mnist.py
python profile_memory.py

# Sweeps
python hyperparam_sweep.py        # Running now

# To implement
python train.py --dataset cifar10
python train.py --local-update    # O(1) memory mode
python plot_figures.py            # Generate paper figures
```

---

## Next Actions (Today)

1. **[NOW]** Let sweep finish → extract best config
2. **[PARALLEL]** Implement `LocalHebbianUpdate` skeleton
3. **[PARALLEL]** Add torch.compile + wandb to train_mnist.py
4. **[AFTER SWEEP]** Validate best config × 5 seeds
5. **[AFTER B1]** Profile memory with local update
6. **[THEN]** Create unified train.py for multi-dataset
