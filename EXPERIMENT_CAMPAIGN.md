# Multi-Track Experimental Campaign

**Goal**: Reach 94% accuracy through comprehensive experimentation  
**Status**: IN PROGRESS  
**Started**: December 29, 2025

---

## Track A: Extended Training ⏳

**Status**: RUNNING (Command ID: 2b1f7048-54a2-443c-80e7-6bcda20af9b6)

```bash
python train.py --d-model 256 --n-heads 8 --d-ff 1024 \
    --beta 0.22 --damping 0.8 --lr 0.002 \
    --epochs 50 --dropout 0.1 --compile
```

**Configuration**:
- β=0.22 (fixed, no annealing)
- 50 epochs (vs 15 baseline)
- All other params identical to β sweep best

**Expected**: 92.37% → 93.5-94%  
**Duration**: ~3.5 hours  
**Log**: Will be streamed to terminal

---

## Track B: Multi-Seed Validation ⏸️

**Status**: READY TO RUN (after extended training completes)

```bash
./run_multiseed_beta022.sh
```

**Plan**: 5 independent runs (seeds 1-5), 30 epochs each  
**Purpose**: Establish statistical significance  
**Expected**: Mean ~92.4%, std < 0.5%  
**Duration**: ~5 hours

---

## Track C: Architecture Scaling ⏸️

**Status**: READY TO RUN

```bash
./run_architecture_scaling.sh
```

**Experiments**:
1. d_model=512, n_heads=16, d_ff=2048 (30 epochs)
2. d_model=256, n_heads=8, d_ff=2048 (30 epochs)

**Expected**: +1.0-1.5% improvement  
**Duration**: ~4 hours total

---

## Track D: Learning Rate Sweep ⏸️

**Status**: READY TO RUN

```bash
./run_lr_sweep.sh
```

**Test**: lr ∈ {0.001, 0.0015, 0.002, 0.0025, 0.003}  
**Duration**: ~3 hours  
**Purpose**: Optimize learning rate for β=0.22

---

## Execution Strategy

### Sequential Execution (Recommended for Single GPU)

1. **Now**: Extended training (50 epochs) - RUNNING
2. **Next**: Multi-seed validation (5 runs) - ~5 hours
3. **Then**: Architecture scaling - ~4 hours
4. **Finally**: Learning rate sweep - ~3 hours

**Total estimated time**: ~15-16 hours

### Parallel Execution (If Multiple GPUs Available)

- GPU 1: Extended training → Multi-seed
- GPU 2: Architecture scaling
- GPU 3: Learning rate sweep

**Total time**: ~8-9 hours

---

## Success Metrics

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Accuracy | 92.37% | 94.0% | 94.5%+ |
| Statistical significance | N/A | ✅ 5-seed validation | ✅ |
| Optimal architecture | d=256 | Found | ✅ |
| Optimal LR | 0.002 | Validated/improved | ✅ |

---

## Monitoring Commands

```bash
# Check extended training progress
tail -f <log_file>  # Will be created when training starts

# Check all running experiments
ps aux | grep python | grep train.py

# Monitor GPU usage
nvidia-smi -l 1
```

---

## Next Actions After Campaign

1. **Analyze all results** - Compare across experiments
2. **Identify best configuration** - Select winning setup
3. **Final validation run** - Confirm reproducibility
4. **Update documentation** - Record findings in docs/05-results.md
5. **Publication preparation** - Create figures and tables

---

**Created**: 2025-12-29 03:37  
**Last updated**: 2025-12-29 03:38
