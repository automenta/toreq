# TorEqProp Phase 2: Scientific Discovery & Novel Contributions

**Strategic Pivot**: Moving beyond "tuning for accuracy" to "characterizing unique advantages." To publish at top venues (NeurIPS/ICML), we must prove not just that EqProp works, but that it possesses desirable properties that BP lacks (robustness, biological plausibility, hardware efficiency) or behaves in scientifically interesting ways (chaos edges, phase transitions).

## 1. Multi-Fidelity & Evolutionary Optimization
**Objective**: Efficiently traverse the vast landscape to find "Islands of Stability" and "Super-Convergence".
- [ ] **Multi-Fidelity Scheduler**: (As originally planned) implementations for resource efficiency.
- [ ] **Population-Based Training (PBT)**: Instead of static trials, evolve hyperparameters during training. Allow low-performing agents to copy params from high-performers and mutate their $\beta$ schedules.
- [ ] **Meta-Learning Schedules**: Learn *dynamic* schedules for $\beta$ and damping, rather than fixed values. Does a "sawtooth" $\beta$ schedule help break out of local minima?

## 2. Dynamical Systems Analysis (The "Why")
**Objective**: Treat EqProp as a physical system. Analyze its behavior to gain theoretical insights BP cannot offer.
- [ ] **Lyapunov Exponent Tracking**: Measure the rate of divergence/convergence during the inference phase. Does "Edge of Chaos" initialization lead to faster learning?
- [ ] **Phase Transition Mapping**: Map the boundary where EqProp transitions from "Convergent" to "Oscillatory/Chaotic".
    - *Hypothesis*: Optimal learning happens near the critical point.
- [ ] **Energy Landscape Topography**: Visualize the implicit energy function $E$. How does learning reshape the basins of attraction?

## 3. Beyond Accuracy: Proving Novel Advantages
**Objective**: Identify where EqProp beats BP *qualitatively*, even if accuracy is matched.
- [ ] **Adversarial Robustness**:
    - *Hypothesis*: The settling process of EqProp naturally filters high-frequency adversarial noise.
    - *Experiment*: Attack both fully-trained models (FGSM/PGD). Measure degradation.
- [ ] **OOD Generalization**:
    - Test on shifted distributions (e.g., MNIST Rotation/Scale). Does the physical intuition of EqProp allow better extrapolation?
- [ ] **Gradient approximations**:
    - Measure cosine similarity between EqProp updates and "True" gradients. When do they diverge? Is the divergence actually *beneficial* (e.g., escaping saddles)?

## 4. Fair & Deep Comparison Toolkit
**Objective**: Rigorous, unassailable evidence of performance characteristics.
- [ ] **Iteration-Budget & FLOP-Budget Matching**: (As planned).
- [ ] **Landscape Geometry**:
    - **Hessian Spectrum Analysis**: Compute eigenvalues of the Hessian at the optimum. Are EqProp minima flatter (= better generalization)?
    - **Mode Connectivity**: Can we linearly interpolate between EqProp and BP solutions without encountering high-loss barriers?

## 5. Statistical & result Rigor
- [ ] **Power Analysis & Confidence Intervals**: (As planned).
- [ ] **Ablation Studies**:
    - "Hard" Ablations: Remove symmetry constraint. How much does it break?
    - "Soft" Ablations: Add noise to the feedback weights. How robust is the learning rule to hardware imperfections (vital for neuromorphic chips)?

## 6. Visualization for Discovery
- [ ] **The "Phase Portrait" of Learning**: Video artifacts showing how fixed points move during training dynamics.
- [ ] **Pareto Frontiers**: Accuracy vs. Energy, Accuracy vs. Robustness.
