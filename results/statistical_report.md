# Statistical Validation Report

**Generated**: 2026-01-02T12:40:12.539603

---
## Benchmark Results

| Model | Mean Acc | Std Dev | Seeds | Status |
|-------|----------|---------|-------|--------|
| BackpropMLP | 97.33% | ±0.48% | 5 | ✅ |
| LoopedMLP (SN) | 95.72% | ±0.22% | 5 | ✅ |
| ModernEqProp (SN) | 95.33% | ±0.94% | 5 | ✅ |

### Significance vs LoopedMLP (SN)

Baseline: **LoopedMLP (SN)** (95.72% ± 0.22%)

| Model | Gap | P-value | Cohen's d | Significance |
|-------|-----|---------|-----------|--------------|
| BackpropMLP | -1.61% | 0.0003 | -4.28 (large) | ✅ Yes |
| ModernEqProp (SN) | 0.39% | 0.4423 | 0.57 (medium) | ⚠️ No |

---

---

## Publishability Assessment

✅ **READY FOR PUBLICATION** - Statistical validation passed