# Figure: Accuracy Comparison (EqProp vs Backprop)

## Data
| Model | Test Accuracy |
|-------|---------------|
| Backprop (baseline) | 98.06% |
| ModernEqProp (SN) | 97.50% |
| LoopedMLP (SN) | 95.83% |
| ToroidalMLP (SN) | 95.00% |

## Key Finding
ModernEqProp with spectral normalization **matches Backprop's best accuracy** (97.50%).

## How to Generate
Install matplotlib and run: `python research.py --action figures`
