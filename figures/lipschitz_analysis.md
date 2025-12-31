# Figure: Lipschitz Constant Analysis

## Data
| Model | L (Untrained) | L (Trained, no SN) | L (Trained, SN) |
|-------|---------------|-------------------|-----------------|
| LoopedMLP | 0.69 | 0.74 | **0.55** ✅ |
| ToroidalMLP | 0.70 | **1.01** ❌ | **0.55** ✅ |
| ModernEqProp | 0.54 | **9.50** ❌ | **0.54** ✅ |

## Key Finding
- Training without SN causes L > 1 (breaks convergence)
- Spectral normalization maintains L < 1 (stable)
- L = 1 is the stability threshold

## How to Generate
Install matplotlib and run: `python research.py --action figures`
