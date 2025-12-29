# TorEqProp Research Plan

> **Single source of truth for all research direction and experimentation.**  
> Last updated: December 29, 2025

---

## 🚀 Quick Start: Turn-Key Discovery

**Run the complete discovery campaign with one command:**

```bash
# Full discovery (all phases, ~4 hours)
python run_discovery.py

# Quick validation (1 epoch each, ~30 min)
python run_discovery.py --quick

# Specific phase only
python run_discovery.py --phase 1        # Dataset sweep (~25 min)
python run_discovery.py --phase 2        # Algorithmic tasks (~25 min)
python run_discovery.py --phase 3        # RL experiments (~25 min)
python run_discovery.py --phase 1 2 3    # Phases 1-3 only

# Preview what would run
python run_discovery.py --dry-run
```

**Results saved to:** `logs/discovery/YYYYMMDD_HHMMSS/`

---

## Executive Summary

Train transformers via Equilibrium Propagation (EqProp), demonstrating gradient equivalence, competitive accuracy, and unique properties. **Prioritize breadth and agility** over depth—identify promising directions quickly, then scale.

### Current Best Results
| Metric | Value | Notes |
|--------|-------|-------|
| MNIST Accuracy | **93.83%** | 50 epochs, β=0.22, d=256 |
| Gradient Equivalence | **0.9972** | cos sim at β=0.001 |
| Optimal β | **0.22** | Fixed (no annealing!) |
| Reproducibility | **±0.26%** | 5-seed validation |

---

## Experiment Infrastructure

### Files Created
| File | Purpose |
|------|---------|
| `run_discovery.py` | **Main orchestrator** - runs all experiments with tracking |
| `train_algorithmic.py` | Algorithmic reasoning tasks (parity, addition, etc.) |
| `train_rl.py` | Reinforcement learning with EqProp |
| `src/datasets.py` | Multi-dataset loader (MNIST, Fashion, CIFAR-10, SVHN, EMNIST) |
| `src/algorithmic_tasks.py` | Task generators for parity, reversal, copy, addition |
| `configs/rapid_mode.yaml` | Fast experimentation defaults |

### Rapid Mode (`--rapid` flag)
```yaml
d_model: 64       # 4× smaller
epochs: 3         # 10× fewer
max_iters: 20     # Faster convergence
```
**Time:** ~5-10 min per experiment

---

## Phase 1: Rapid Dataset Sweep

**Goal:** Characterize TorEqProp across diverse domains in <30 min total.  
**Run:** `python run_discovery.py --phase 1`

### Experiments
| # | Dataset | Command | Time | Success Threshold |
|---|---------|---------|------|-------------------|
| 1 | MNIST | `python train.py --dataset mnist --rapid --epochs 3` | 5 min | ≥80% |
| 2 | FashionMNIST | `python train.py --dataset fashion --rapid --epochs 3` | 5 min | ≥70% |
| 3 | CIFAR-10 | `python train.py --dataset cifar10 --rapid --epochs 3` | 8 min | ≥35% |
| 4 | SVHN | `python train.py --dataset svhn --rapid --epochs 3` | 8 min | ≥40% |

### Manual Execution
```bash
# Run all datasets sequentially
for ds in mnist fashion cifar10 svhn; do
  echo "=== $ds ==="
  python train.py --dataset $ds --rapid --epochs 3 2>&1 | tee logs/rapid_${ds}.log
done
```

### Success Criteria
- [ ] All datasets train without errors
- [ ] Each achieves >random baseline accuracy
- [ ] Identify 2 datasets besides MNIST for deeper exploration

### Interpretation Guide
| Accuracy | Meaning | Action |
|----------|---------|--------|
| >80% | Strong signal | Scale up (more epochs, larger model) |
| 60-80% | Moderate | Tune hyperparameters |
| 40-60% | Weak | May need architecture changes |
| <40% | Minimal | Investigate or skip |

---

## Phase 2: Algorithmic Reasoning Tasks

**Goal:** Test adaptive compute hypothesis—do harder instances need more iterations?  
**Run:** `python run_discovery.py --phase 2`

### Experiments
| # | Task | Command | Time | Threshold |
|---|------|---------|------|-----------|
| 1 | Parity N=8 | `python train_algorithmic.py --task parity --seq-len 8 --epochs 10` | 5 min | ≥90% |
| 2 | Parity N=12 | `python train_algorithmic.py --task parity --seq-len 12 --epochs 15` | 8 min | ≥85% |
| 3 | Copy | `python train_algorithmic.py --task copy --seq-len 8 --epochs 5` | 3 min | ≥95% |
| 4 | Addition 4-digit | `python train_algorithmic.py --task addition --n-digits 4 --epochs 20` | 10 min | ≥50% |

### The Adaptive Compute Hypothesis
```
If equilibrium models allocate compute dynamically:
  - Harder parity instances (more 1s) → more iterations
  - Longer sequences → more iterations
  - More carries in addition → more iterations
```

### Analysis Commands
```bash
# Run with iteration tracking
python train_algorithmic.py --task parity --seq-len 8 --epochs 10 --analyze-difficulty

# Check iteration variance
# High variance (>0.5 std) = adaptive compute detected
# Low variance = uniform convergence (no advantage)
```

### Success Criteria
- [ ] Achieve >90% on parity (N=8)
- [ ] Measure per-sample iteration counts
- [ ] Find correlation between difficulty and iterations
- [ ] **PUBLISHABLE IF:** correlation R² > 0.5

---

## Phase 3: Reinforcement Learning

**Goal:** Novel application—use EqProp for policy gradient estimation.  
**Run:** `python run_discovery.py --phase 3`

### Experiments
| # | Experiment | Command | Time | Threshold |
|---|------------|---------|------|-----------|
| 1 | CartPole EqProp | `python train_rl.py --env CartPole-v1 --episodes 500` | 15 min | ≥195 avg |
| 2 | CartPole BP | `python train_rl.py --env CartPole-v1 --episodes 500 --use-bp` | 10 min | ≥195 avg |

### Comparison Protocol
```bash
# Run both policies
python train_rl.py --env CartPole-v1 --episodes 500 | tee logs/rl/cartpole_eqprop.log
python train_rl.py --env CartPole-v1 --episodes 500 --use-bp | tee logs/rl/cartpole_bp.log

# Compare:
# 1. Episodes to solve (avg reward ≥195)
# 2. Final average reward
# 3. Training stability (reward variance)
```

### Success Criteria
- [ ] Solve CartPole (≥195 avg reward over 100 episodes)
- [ ] Compare convergence speed: EqProp vs BP
- [ ] **PUBLISHABLE IF:** EqProp matches or exceeds BP

### Prerequisites
```bash
pip install gymnasium  # Required for RL experiments
```

---

## Phase 4: Accuracy Push to 95%

**Goal:** Maximize MNIST accuracy with proven configuration.  
**Run:** `python run_discovery.py --phase 4`

> **Note:** Only run after rapid exploration in Phases 1-3.

### Experiments
| # | Config | Command | Time | Target |
|---|--------|---------|------|--------|
| 1 | Extended (100 epochs) | `python train.py --d-model 256 --beta 0.22 --epochs 100 --dropout 0.1 --compile` | 3 hours | ≥94.5% |
| 2 | Scaled (d=512) | `python train.py --d-model 512 --n-heads 16 --d-ff 2048 --beta 0.22 --epochs 50 --compile` | 4 hours | ≥95% |

### Known Trajectory
```
Epochs →  15      30      50      100
Accuracy: 92.37%  ~93%    93.83%  ~94.5%?
```

### Success Criteria
- [ ] Reach 94.5%+ on single run
- [ ] Validate with 3-seed average
- [ ] **PUBLISHABLE THRESHOLD:** 95%

---

## Phase 5: O(1) Memory Verification

**Goal:** Demonstrate constant memory regardless of equilibrium iterations.  
**Run:** `python run_discovery.py --phase 5`

### Experiments
| # | Model Size | Command | Time | Target Ratio |
|---|------------|---------|------|--------------|
| 1 | d=256 | `python profile_memory.py --d-model 256 --max-iters 100` | 5 min | <1.5× |
| 2 | d=1024 | `python profile_memory.py --d-model 1024 --max-iters 100` | 10 min | <0.8× |
| 3 | d=2048 | `python profile_memory.py --d-model 2048 --max-iters 100` | 20 min | <0.5× |

### Expected Behavior
```
      Memory
        ↑
   BP   |        /
        |      /
        |    /
        |  /
EqProp  |__________→ d_model
        
BP: Memory ∝ depth (stores all activations)
EqProp: Memory = O(1) (only current state)
```

### Success Criteria
- [ ] Memory ratio (EqProp/BP) < 0.5 at d=2048
- [ ] Generate publication-ready figure
- [ ] **PUBLISHABLE IF:** Clear constant-memory scaling

---

## Critical Insights (From Prior Experiments)

### What Works ✅
| Setting | Value | Evidence |
|---------|-------|----------|
| β | **0.22 fixed** | Best accuracy, stable across 50 epochs and 5 seeds |
| d_model | **256** | Good capacity, fast training |
| Epochs | **50+** | 93.83% at 50 epochs, still improving |
| Mode | **Non-symmetric** | Works better than symmetric |
| Attention | **Linear** | Required for symmetry claims |

### What Doesn't Work ❌
| Setting | Problem |
|---------|---------|
| β-annealing | Causes catastrophic collapse |
| β < 0.20 | Marginal gains, risk not worth it |
| Symmetric softmax | Unstable gradients |
| Short training (<15 epochs) | Insufficient convergence |

### Theory-Practice Gaps
| Theory Says | Practice Shows | Publication Value |
|-------------|----------------|-------------------|
| β→0 best | β=0.22 best | HIGH - novel finding |
| Symmetric required | Non-symmetric works | MEDIUM |
| O(1) memory | 1.06× BP (at d=256) | Needs scale to verify |

---

## Publication Targets

### Primary Claims (Validated ✅)
1. **First transformer trained via EqProp** — 93.83% MNIST
2. **β>0 stability threshold** — Novel finding contradicting theory
3. **Reproducibility** — ±0.26% across 5 seeds

### Stretch Claims (Need Validation)
4. **Adaptive compute** — Requires algorithmic task correlation
5. **RL with equilibrium** — Requires CartPole success
6. **O(1) memory** — Requires d=2048 profiling

### Venues
| Venue | Requirements | Deadline |
|-------|--------------|----------|
| ICML 2025 | 2+ validated claims | Feb 2025 |
| NeurIPS 2025 | 2+ validated claims | May 2025 |
| ICLR 2025 Workshop | 1 validated claim | Oct 2024 |

---

## Timeline

| Day | Focus | Commands | Expected Output |
|-----|-------|----------|-----------------|
| 1 | Rapid sweep (Phases 1-3) | `python run_discovery.py --phase 1 2 3` | Results on 8+ experiments |
| 2 | Analyze & iterate | Review `logs/discovery/` | Promising directions identified |
| 3 | Scale promising | Phase 4 on best candidates | 94%+ accuracy |
| 4 | Memory & RL depth | Phase 5 + deeper RL | O(1) demo, RL comparison |
| 5 | Documentation | Update README, create figures | Publication-ready |

---

## File Organization

```
toreq/
├── TODO.md                    # THIS FILE (single source of truth)
├── run_discovery.py           # 🔑 Main experiment orchestrator
├── train.py                   # Classification training
├── train_algorithmic.py       # Algorithmic tasks training
├── train_rl.py                # RL training
├── src/
│   ├── datasets.py            # Multi-dataset loader
│   ├── algorithmic_tasks.py   # Task generators
│   ├── config.py              # Config with --rapid flag
│   ├── models.py              # LoopedTransformerBlock
│   ├── solver.py              # EquilibriumSolver
│   ├── trainer.py             # EqPropTrainer
│   └── updates.py             # Update strategies
├── configs/
│   └── rapid_mode.yaml        # Fast experimentation defaults
├── logs/
│   └── discovery/             # Experiment results
└── docs/                      # Reference documentation only
```

---

## Deprecated Documents

> These files are archived and superseded by this TODO.md:

- `EXPERIMENT_CAMPAIGN.md`
- `docs/NEXT_EXPERIMENTS.md`
- `docs/06-research-roadmap.md` (reference only)
- `TRACK_A_RESULTS.md` (historical record only)

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size
python train.py --rapid --batch-size 64

# Or use CPU
python train.py --rapid --device cpu
```

**2. gymnasium not installed (RL)**
```bash
pip install gymnasium
```

**3. Dataset download fails**
```bash
# Manual download to ./data
# Or specify different directory
python train.py --data-dir /path/to/data
```

**4. Slow training**
```bash
# Enable torch.compile
python train.py --compile

# Use rapid mode for exploration
python train.py --rapid
```

---

## Appendix: All Experiment Commands

### Phase 1: Dataset Sweep
```bash
python train.py --dataset mnist --rapid --epochs 3
python train.py --dataset fashion --rapid --epochs 3
python train.py --dataset cifar10 --rapid --epochs 3
python train.py --dataset svhn --rapid --epochs 3
```

### Phase 2: Algorithmic Tasks
```bash
python train_algorithmic.py --task parity --seq-len 8 --epochs 10
python train_algorithmic.py --task parity --seq-len 12 --epochs 15
python train_algorithmic.py --task copy --seq-len 8 --epochs 5
python train_algorithmic.py --task addition --n-digits 4 --epochs 20
```

### Phase 3: Reinforcement Learning
```bash
python train_rl.py --env CartPole-v1 --episodes 500
python train_rl.py --env CartPole-v1 --episodes 500 --use-bp
```

### Phase 4: Accuracy Push
```bash
python train.py --d-model 256 --beta 0.22 --epochs 100 --dropout 0.1 --compile
python train.py --d-model 512 --n-heads 16 --d-ff 2048 --beta 0.22 --epochs 50 --compile
```

### Phase 5: Memory Profiling
```bash
python profile_memory.py --d-model 256 --max-iters 100
python profile_memory.py --d-model 1024 --max-iters 100
python profile_memory.py --d-model 2048 --max-iters 100
```
