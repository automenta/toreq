# TorEqProp Verification Results

**Generated**: 2026-01-04 18:57:18


## Executive Summary

**Verification completed in 4.4 seconds.**

### Overall Results

| Metric | Value |
|--------|-------|
| Tracks Verified | 1 |
| Passed | 1 ✅ |
| Partial | 0 ⚠️ |
| Failed | 0 ❌ |
| Stubs (TODO) | 0 🔧 |
| Average Score | 100.0/100 |

### Track Summary

| # | Track | Status | Score | Time |
|---|-------|--------|-------|------|
| 1 | Spectral Normalization Stability | ✅ | 100 | 4.4s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 1: Spectral Normalization Stability


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 4.4s


**Claim**: Spectral normalization constrains Lipschitz constant L ≤ 1, unlike unconstrained training.

**Experiment**: Train identical networks with and without spectral normalization.

| Configuration | L (before) | L (after) | Δ | Constrained? |
|---------------|------------|-----------|---|--------------|
| Without SN | 0.975 | 14.816 | +13.84 | ❌ No |
| With SN | 1.014 | 1.001 | -0.01 | ✅ Yes |

**Key Difference**: L(no_sn) - L(sn) = 13.815

**Interpretation**: 
- Without SN: L = 14.82 (unconstrained, can grow)
- With SN: L = 1.00 (constrained to ~1.0)
- SN provides 1381% reduction in Lipschitz constant


