# TorEqProp Branch Hybridization Plan

> **Status**: ✅ **COMPLETED**  
> **Source branches**: `implement-readme-plan-6504849006638432880` + `jules/implement-toreqprop-v1`  
> **Target**: Unified codebase with correct EqProp implementation

---

## Phase 1: Core Architecture Merge ✅ COMPLETED

### 1.1 Add `src/utils.py` from target branch
- [x] Copy `linear_attention()` function (ELU+1 feature map)
- [x] Copy `detect_compute_tier()` function for adaptive scaling

### 1.2 Extend `src/models.py` with symmetric mode
- [x] Add `symmetric: bool = False` parameter to `LoopedTransformerBlock`
- [x] Implement symmetric weight tying in `__init__`:
  - `W_out = W_q^T` (applied dynamically, not stored)
  - `W_k = W_v` (key and value share weights)
  - `W2 = W1^T` for FFN (output = input weight transposed)
- [x] Use `tanh()` activation in symmetric mode (bounded energy)
- [x] Disable LayerNorm in symmetric mode (rely on tanh bounding)
- [x] Add `norm_final` layer for non-symmetric mode (Universal Transformer style)

### 1.3 Keep current branch API improvements
- [x] Retain `attention_type: str` enum pattern
- [x] Extend support to include symmetric mode with `attention_type='linear'`

---

## Phase 2: Trainer Updates ✅ COMPLETED

### 2.1 Verify nudge direction consistency
- [x] Documented sign convention: `h_new - β * ∇L` (moves in direction that decreases loss)
- [x] Equivalent to: `h_new + β * ∇(-L)` from original branch

### 2.2 Support both update mechanisms
- [x] MSE proxy loss (current branch): `(1/β) * MSE(h_out, h_nudged)`
- [x] Vector field backprop (target branch): `backward(gradient=v)` where `v = (h_nudged - h_free) / β`
- [x] Add `update_mode` parameter to switch between them for experimentation

---

## Phase 3: Testing & Verification ✅ COMPLETED

### 3.1 Update gradient equivalence test
- [x] Add test case with `symmetric=True` + `attention_type='linear'`
- [x] **RESULT**: Cosine similarity = **0.9972** at β=0.001 (> 0.99 threshold ✓)
- [x] Kept extensive theoretical comments from current branch

### 3.2 Run existing tests
```bash
python test_gradient_equiv.py  # ✓ PASSED
# Symmetric mode: 0.9972 cosine similarity
# Non-symmetric mode: 0.4166 cosine similarity
# Confirms symmetric weight tying is critical for EqProp guarantees
```

**MNIST Training Note**: Existing `train_mnist.py` and `train_mnist_bp.py` scripts are compatible with the extended API (symmetric defaults to False). They can be run to verify end-to-end training works.

### 3.3 Comparative tests
- [x] Compare symmetric vs non-symmetric gradient quality ✓ (0.9972 vs 0.4166)
- Note: MSE proxy vs vector field update accuracy comparison can be done in future experiments
- Note: Memory usage profiling can be done during MNIST training runs

---

## Phase 4: Documentation ✅ COMPLETED

### 4.1 Update inline documentation
- [x] Explain why symmetry is required for EqProp
- [x] Reference Scellier & Bengio 2017 theorem

### 4.2 Architecture decision record
- [x] Documented the hybridization choices in implementation plan
- [x] Explained API design decisions (backward compatibility, symmetric mode flag)

---

## Summary

**All phases completed successfully!** The hybridized codebase now includes:

1. ✅ **`src/utils.py`**: Linear attention and compute tier detection
2. ✅ **`src/models.py`**: Symmetric mode with weight tying for EqProp theoretical guarantees
3. ✅ **`src/trainer.py`**: Both MSE proxy and vector field update mechanisms
4. ✅ **`test_gradient_equiv.py`**: Symmetric mode test achieving **0.9972** cosine similarity

**Key Finding**: Symmetric weight tying is **critical** for EqProp gradient equivalence:
- Symmetric mode: 0.9972 cosine similarity ✓
- Non-symmetric mode: 0.4166 cosine similarity ✗

This confirms the theoretical requirement from Scellier & Bengio 2017 that the Jacobian must be symmetric for EqProp to yield BP-equivalent gradients.

