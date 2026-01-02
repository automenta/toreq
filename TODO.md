# TorEqProp: Research Roadmap

> **Vision**: Prove that Equilibrium Propagation can match—and exceed—backpropagation on modern architectures, opening the door to neuromorphic AI.

---

## ✅ Completed Milestones

### Kernel Implementation
- **Pure NumPy/CuPy kernel** — 1,056 lines, zero PyTorch dependency
- **58% faster than PyTorch** (21.4ms vs 33.9ms with aggressive settings)
- **2.49x GPU speedup** over CPU via CuPy
- **MNIST learning confirmed** — 69% accuracy in 5 epochs

### Core Discoveries
- **Spectral normalization is essential** — L reduced from 21.1 → 0.58
- **Fixed β = 0.22 optimal** — annealing causes collapse
- **Equilibrium solving is the bottleneck** — 78% of training time

---

## 🎯 Phase 1: Complete CPU/GPU Research

**Goal**: Generate undeniable evidence that EqProp is a viable, competitive alternative to backpropagation.

### 1.1 Accuracy Validation *(Target: 1 week)*

| Task | Status | Target |
|------|--------|--------|
| Multi-seed MNIST (5 seeds) | ⬜ | 95%+ with kernel |
| Fashion-MNIST benchmark | ⬜ | 90%+ |
| Hierarchical CIFAR-10 | ⬜ | 60%+ with EnhancedMSTEP |
| Kernel vs PyTorch accuracy parity | ⬜ | Within 1% |

**Key Question**: Does fixing max_steps=8 affect final accuracy?

### 1.2 Speed Validation *(Target: 3 days)*

| Task | Status | Target |
|------|--------|--------|
| Profiled training comparison | ✅ | Kernel ≤ PyTorch |
| Memory scaling test (depth) | ⬜ | O(1) verified |
| Throughput benchmarks | ⬜ | 5000+ samples/sec |

### 1.3 Ablation Studies *(Target: 3 days)*

| Task | Status | Purpose |
|------|--------|---------|
| With/without spectral norm | ⬜ | Prove necessity |
| β sweep (0.15 → 0.30) | ⬜ | Characterize stability region |
| max_steps sweep (5 → 25) | ⬜ | Accuracy-speed tradeoff |
| Damping γ sweep | ⬜ | Convergence analysis |

---

## 📚 Phase 2: Organize for Outreach

**Goal**: Package findings into compelling, reproducible, and publication-ready artifacts.

### 2.1 Publication Figures *(Target: 2 days)*

- [ ] Training curves (EqProp vs Backprop)
- [ ] Lipschitz evolution during training
- [ ] Kernel speedup comparison chart
- [ ] Memory scaling plot

### 2.2 Paper Draft *(Target: 1 week)*

**Paper A: Spectral Normalization for EqProp Stability**
- Update with kernel performance data
- Add multi-seed validation results
- Target: NeurIPS 2025 / ICML 2025

### 2.3 Code Release *(Target: 3 days)*

- [ ] Clean kernel API documentation
- [ ] Example notebooks
- [ ] Reproduction scripts
- [ ] Docker container for benchmarks

### 2.4 Community Engagement *(Target: ongoing)*

- [ ] arXiv preprint (to timestamp novelty)
- [ ] Reddit/X announcement with key graph
- [ ] GitHub repo polish (stars, issues enabled)

---

## 🔮 Phase 3: Hardware Deployment *(Deferred)*

> After CPU/GPU research is validated and published.

### 3.1 FPGA Integration
- Convert kernel to HLS (Vitis)
- Target: Kria KV260 at <10mW
- Measure real power consumption

### 3.2 Neuromorphic Exploration
- Apply for Intel INRC (Loihi 2)
- Adapt kernel for spiking dynamics
- Test on temporal tasks

---

## 📊 Evidence Tracker

| Claim | Evidence Status | Confidence |
|-------|-----------------|------------|
| Spectral norm guarantees L < 1 | ✅ 3 seeds | 95% |
| EqProp matches backprop accuracy | ✅ 1 seed | 90% → *needs 5 seeds* |
| β=0.22 optimal for stability | ✅ 7-value sweep | 88% |
| O(1) memory training | ⚠️ Incomplete | 30% → *priority* |
| Kernel faster than PyTorch | ✅ Profiled | 95% |

---

## 🚀 Quick Wins (Do Today)

```bash
# 1. Run multi-seed validation
python scripts/competitive_benchmark.py --seeds 5

# 2. Test hierarchical CIFAR-10
python scripts/test_cifar_readiness.py --model EnhancedMSTEP --epochs 20

# 3. Verify memory scaling
python scripts/validate_o1_memory.py
```

---

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| MNIST accuracy (kernel) | 69% | 95% | 🟡 |
| CIFAR-10 accuracy | 19.9% | 60% | 🔴 |
| Kernel speed vs PyTorch | 1.04x | ≤1.0x | ✅ |
| arXiv preprint | — | Submitted | 🔴 |
| Conference submission | — | NeurIPS 2025 | 🔴 |

---

*Last updated: 2026-01-02*