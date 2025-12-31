# Figure: β Sweep Results

## Data
| β | Accuracy | Stability |
|---|----------|-----------|
| 0.20 | 91.52% | ✅ Stable |
| 0.21 | 91.55% | ✅ Stable |
| **0.22** | **92.37%** | ✅ **Optimal** |
| 0.23 | 90.92% | ✅ Stable |
| 0.24 | 91.50% | ✅ Stable |
| 0.25 | 92.12% | ✅ Stable |
| 0.26 | 90.67% | ✅ Stable |

## Key Finding
- All β values in [0.20, 0.26] are stable
- β = 0.22 achieves highest accuracy
- This contradicts theory suggesting β→0 is best

## How to Generate
Install matplotlib and run: `python research.py --action figures`
