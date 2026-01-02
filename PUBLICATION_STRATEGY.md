# TorEqProp: Multi-Paper Publication Strategy

> **Goal**: Maximize research impact through strategic paper splitting and target venue matching.

---

## Publication Portfolio (4 Papers)

### Paper A: Spectral Normalization Enables Stable EqProp ⭐ **FLAGSHIP**
**Target**: NeurIPS 2025 / ICML 2025 (Main Track)  
**Status**: Template ready, needs experimental validation  
**Timeline**: 2-3 weeks

**Core Contribution**: First demonstration that spectral normalization solves EqProp stability, enabling competitive accuracy (97.50% = backprop).

**Key Results Needed**:
- ✅ Lipschitz L < 1 maintained during training
- ⬜ Multi-seed MNIST ≥ 94% (5 seeds)
- ⬜ CIFAR-10 ≥ 50% (hierarchical)
- ⬜ Ablation: without SN → divergence

**Novelty**: First rigorous EqProp scaling to modern architectures.

---

### Paper B: Fixed β Beats Annealing
**Target**: TMLR / JMLR / NeurIPS Workshop  
**Status**: Template exists, needs multi-seed data  
**Timeline**: 1-2 weeks

**Core Contribution**: Empirical discovery that β-annealing causes collapse; fixed β=0.22 is optimal.

**Key Results Needed**:
- ⬜ β-annealing collapse reproduced (3+ seeds)
- ⬜ β sweep [0.15-0.30] with 5-7 values
- ⬜ Optimal β characterization
- ✅ Stability range [0.20-0.26] validated

**Novelty**: First systematic β study; contradicts theory (β→0 better).

**Publication Strategy**: 
- **Option 1**: Standalone paper (TMLR, 4-6 months review)
- **Option 2**: Appendix to Paper A (stronger combination)
- **Recommendation**: Include in Paper A for maximum impact

---

### Paper C: A Pure NumPy/CuPy Kernel for Portable EqProp ⭐ **SYSTEMS**
**Target**: MLSys 2026 / NeurIPS Systems Track  
**Status**: Implementation complete, needs writeup  
**Timeline**: 2 weeks

**Core Contribution**: Standalone, autograd-free EqProp kernel achieving 58% speedup over PyTorch.

**Key Results**:
- ✅ Kernel implementation (1,056 lines)
- ✅ 58% faster than PyTorch (21.4ms vs 33.9ms, aggressive mode)
- ✅ 2.49x GPU speedup
- ✅ 69% MNIST accuracy (learning confirmed)
- ⬜ CIFAR-10 validation
- ⬜ Memory scaling O(1) verification

**Novelty**: First portable, hardware-deployable EqProp implementation. Directly translates to HLS/FPGA.

**Why This Matters**:
- Enables neuromorphic deployment
- Proves EqProp viability for edge AI
- Reference implementation for researchers

**Publication Strategy**: 
- Strong fit for **MLSys** (systems/implementation focus)
- Alternative: **JMLR** Open Source Software track
- Can cite Paper A for theoretical foundation

---

### Paper D: Hierarchical EqProp for Vision Tasks
**Target**: ICLR 2026 / Computer Vision venue  
**Status**: Models exist (EnhancedMSTEP), needs experimental validation  
**Timeline**: 4-6 weeks *(contingent on CIFAR-10 success)*

**Core Contribution**: Multi-scale hierarchical architectures enable EqProp scaling to complex vision tasks.

**Key Results Needed**:
- ⬜ CIFAR-10 ≥ 60% with EnhancedMSTEP
- ⬜ Ablation: hierarchical vs flat
- ⬜ ImageNet-32 proof-of-concept (optional)

**Novelty**: First hierarchical EqProp architecture; first serious CIFAR-10 results.

**Publication Strategy**: 
- **Contingent**: Only if CIFAR-10 results are strong (≥50%)
- If results weak (<50%), **defer to future work** or include as preliminary in Paper A

---

## Publication Timeline

```
2026-01-02 (Now)
    ↓
    ├── Week 1-2: Run Experiments (Phases 1-2 from TODO.md)
    │   ├── Multi-seed MNIST
    │   ├── Hierarchical CIFAR-10
    │   ├── Ablations
    │   └── Speed/memory validation
    ↓
    ├── Week 3: Generate Papers A+B
    │   ├── Run generate_paper.py for Paper A
    │   ├── Write Paper C (kernel) manually
    │   └── Validate all claims
    ↓
    ├── Week 4: arXiv Submission
    │   ├── Paper A (Spectral Norm) → arXiv ⭐ PRIORITY
    │   ├── Paper C (Kernel) → arXiv
    │   └── Community announcement
    ↓
    ├── Weeks 5-6: Conference Preparation
    │   ├── Paper A → NeurIPS 2025 (May deadline)
    │   └── Paper C → MLSys 2026 (Oct deadline)
    ↓
    ├── Contingent: Paper D
        └── Only if CIFAR-10 ≥ 50% by Week 2
```

---

## Recommended Submission Strategy

### Immediate (Next 2 months)

1. **Paper A (Spectral Norm)** → NeurIPS 2025
   - Strongest contribution
   - Include β-annealing discovery as secondary finding
   - Target main conference track

2. **Paper C (Kernel)** → arXiv + MLSys 2026
   - Systems contribution, different audience
   - Enables reproducibility
   - Cite Paper A for theory

### Medium-term (3-6 months)

3. **Paper D (Hierarchical)** → ICLR 2026 *(if CIFAR-10 strong)*
   - Only pursue if ≥60% accuracy achieved
   - Otherwise include as "future work" in Paper A

### Long-term (Post-publication)

4. **Paper E: Neuromorphic Deployment** *(deferred to Phase 3)*
   - FPGA/Loihi implementation
   - Real power measurements
   - Target: Nature Electronics / specialized venue

---

## Bundling vs Splitting Decision Matrix

| Approach | Papers | Pros | Cons | Recommendation |
|----------|--------|------|------|----------------|
| **Bundle All** | 1 paper (A+B+C) | Comprehensive, high impact | Dilutes focus, harder review | ❌ Too much |
| **Split A+B, Separate C** | 2 papers | Balanced, clear narrative | β discovery might be weak alone | ✅ **BEST** |
| **Split All** | 3-4 papers | More publications | Salami slicing, lower impact each | ⚠️ Risky |
| **A only** | 1 flagship | Maximum focus | Leaves kernel unpublished | ❌ Wastes work |

### Final Recommendation: **2-Paper Strategy**

1. **Paper A**: Spectral Norm + β-Annealing (NeurIPS/ICML)
   - Combines two complementary findings
   - Stronger experimental section
   - More compelling narrative

2. **Paper C**: Kernel Implementation (MLSys/JMLR-OSS)
   - Different venue, different audience
   - Systems contribution
   - Enables adoption

3. **Paper D**: Contingent (defer decision to Week 2 results)

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| MNIST < 94% | High | Extend training, tune hyperparameters, or adjust claim to "competitive" |
| CIFAR-10 < 35% | Medium | Drop Paper D, include as preliminary in Paper A appendix |
| Kernel O(1) fails | Low | Report as "theoretical" + "implementation in progress" |
| NeurIPS rejection | Medium | Resubmit to ICML or pivot to TMLR (faster acceptance) |
| Scooped on spectral norm | High | Rush arXiv submission to timestamp priority |

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Papers submitted | ≥ 2 | ⬜ 0/2 |
| arXiv preprints | ≥ 2 | ⬜ 0/2 |
| Conference acceptances | ≥ 1 | ⬜ 0/1 |
| GitHub stars | ≥ 100 | ⬜ TBD |
| Community citations | ≥ 5 | ⬜ 0/5 |

---

## Next Actions (This Week)

```bash
# 1. Run complete experimental pipeline
./run_complete_research.sh

# 2. Generate Paper A draft
python scripts/generate_paper.py --paper spectral_normalization

# 3. Start Paper C writeup
cp papers/paper_template.md papers/kernel_paper.md
# Edit manually with kernel results

# 4. Validate all claims
python scripts/validate_claims.py

# 5. Decision point: Paper D
if [ CIFAR10_ACC -ge 50 ]; then
    echo "Proceed with Paper D"
else
    echo "Defer Paper D to future work"
fi
```

---

*Created: 2026-01-02*
