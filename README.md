### Toroidal Equilibrium Propagation for Transformers (TorEqProp)

#### Overview
**Toroidal Equilibrium Propagation (TorEqProp)** is a novel training paradigm that combines **looped (or toroidal/recurrent-depth) transformers** with **Equilibrium Propagation (EqProp)** to achieve fully symmetric, biologically plausible credit assignment. By "looping the transformer into a torus" — recursively feeding outputs back as inputs with weight-tying — the feedforward transformer becomes a convergent recurrent network capable of relaxing to fixed-point equilibria. EqProp then trains this looped architecture using only forward-phase relaxations (free and nudged equilibria), eliminating the asymmetric backward pass of backpropagation (BP).

This approach directly addresses the original intuition: equalizing forward (inference) and backward (learning) signal metrics through identical recurrent dynamics. Both phases use the same looped forward computations, ensuring balanced propagation of activations and gradients.

The key novelty lies in the direct application of EqProp's contrastive Hebbian mechanism to toroidal transformers, yielding a scalable, local, hardware-friendly alternative to BP for transformer-scale models — distinct from existing energy-based transformers (which rely on implicit differentiation) or DEQs (which use root-finding with BP-like gradients).

#### Architecture: Looped/Toroidal Transformer
- Start with a standard transformer block \( f_\theta(h; x) \), where \( h \) is the hidden state and \( x \) is the input sequence (with positional encodings if needed).
- **Toroidal looping**: Recursively apply the same block:
  \[
  h_{t+1} = h_t + f_\theta(h_t; x) \quad \text{(residual version for stability)}
  \]
  or simply \( h_{t+1} = f_\theta(h_t; x) \).
- This creates a weight-tied recurrent network with toroidal topology (closed loop, no start/end).
- **Inference (free phase)**: Iterate until convergence to equilibrium \( h^* \) where \( h^* \approx f_\theta(h^*; x) \) (or residual form \( 0 \approx f_\theta(h^*; x) \)).
  - Use fixed-point iteration, Broyden's method, or Anderson acceleration for fast convergence.
  - Adaptive loops: More iterations for complex inputs (System 2-like "thinking").
- The equilibrium \( h^* \) represents the processed representation/prediction.

This is inspired by looped transformers (e.g., Universal Transformers, recent 2024–2025 works on expressive power and reasoning) but optimized for convergence rather than finite unrolling.

#### Training: Equilibrium Propagation on the Torus
EqProp trains energy-based models via two symmetric relaxation phases:

1. **Free phase**: Relax to equilibrium \( h^* \) with input \( x \) clamped (no target nudge). This minimizes an implicit energy \( E(h; \theta) \).
2. **Nudged phase**: Slightly perturb output neurons toward the target \( y \) with small factor \( \beta > 0 \) (e.g., add \( \beta (y - h_L) \) to final layer dynamics, where \( h_L \) is output projection). Relax to nearby equilibrium \( h^\beta \).
3. **Weight update**: Contrastive Hebbian rule
   \[
   \Delta \theta \propto (h^\beta (h^\beta)^T - h^* (h^*)^T)
   \]
   (or per-synapse co-activations; local and Hebbian).

- In the limit \( \beta \to 0 \), this exactly matches BP gradients for the fixed-point system.
- Both phases use identical toroidal dynamics — fully equalized forward/backward signals.
- No separate backward pass, no weight transport problem, highly local updates.

#### Energy Formulation (Optional but Recommended)
Frame the looped transformer as minimizing an energy:
\[
E(h; \theta, x) = -\frac{1}{2} h^T W h + \text{other terms}
\]
Dynamics: \( \dot{h} = - \frac{\partial E}{\partial h} \), leading to Hopfield-like convergence.

#### Advantages
- **Biological plausibility**: Symmetric dynamics, local Hebbian updates, no asymmetric BP.
- **Efficiency**: O(1) memory (no unrolling), adaptive compute at inference.
- **Scalability**: Leverages transformer's expressiveness; potential for better reasoning via deeper equilibria.
- **Hardware-friendly**: Suitable for neuromorphic chips (symmetric recurrent relaxations).
- **Generalization**: Equilibria enable natural uncertainty estimation and iterative refinement.

#### Implementation Guidelines
1. **Base block**: Use a standard transformer layer (multi-head attention + FFN) with residuals/LayerNorm for convergence.
2. **Relaxation solver**: Broyden's method (fast, low-memory) or simple fixed-point iteration with damping.
3. **Nudging**: For sequence tasks, nudge final tokens or a classifier head on \( h^* \).
4. **Stabilization**: Jacobian regularization, timestep encodings (from looped transformer literature).
5. **Tasks**: Start with classification (MNIST/CIFAR as recurrent fixed-point), then language modeling (predict next token from equilibrium).

#### Potential Challenges & Mitigations
- Convergence: Ensure contraction (spectral norm <1); use residuals.
- Scaling β: Small β for exact BP approximation; schedule during training.
- Sequential data: Inject input persistently; use causal masking.

This specification provides a complete blueprint for researching/implementing TorEqProp — a genuine post-BP paradigm merging the best of looped transformers and equilibrium learning.