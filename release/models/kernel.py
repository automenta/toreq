"""
EqProp Kernel: Pure NumPy implementation (no autograd)

This implements "true" Equilibrium Propagation using contrastive Hebbian learning
without relying on PyTorch autograd. Demonstrates the O(1) memory property.

Key features:
- No computational graph stored (O(1) memory)
- Manual gradient computation via equilibrium difference
- Power iteration spectral normalization
- Works with only NumPy (optional CuPy for GPU)
"""

import numpy as np
from typing import Dict, Tuple, Optional


def spectral_normalize(W: np.ndarray, u: np.ndarray = None, n_iters: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Power iteration spectral normalization.
    
    Normalizes W by its largest singular value.
    Returns: (W_normalized, u_new, sigma)
    """
    if u is None:
        u = np.random.randn(W.shape[0]).astype(np.float32)
        u = u / np.linalg.norm(u)
    
    for _ in range(n_iters):
        # v = W^T u / ||W^T u||
        v = W.T @ u
        v = v / (np.linalg.norm(v) + 1e-8)
        
        # u = W v / ||W v||
        u = W @ v
        u = u / (np.linalg.norm(u) + 1e-8)
    
    # sigma = u^T W v
    sigma = float(u @ W @ v)
    
    # Normalize W
    W_norm = W / (sigma + 1e-8)
    
    return W_norm, u, sigma


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss from logits."""
    probs = softmax(logits)
    batch_size = logits.shape[0]
    log_probs = np.log(probs[np.arange(batch_size), targets] + 1e-8)
    return -np.mean(log_probs)


class EqPropKernel:
    """
    Pure NumPy Equilibrium Propagation kernel.
    
    Implements true EqProp:
    1. Free phase: iterate to equilibrium h*
    2. Nudged phase: perturb toward target, find h_β
    3. Hebbian update: ΔW ∝ (h_β ⊗ h_β - h* ⊗ h*) / β
    
    This is O(1) memory because we only store current state, not computational graph.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gamma: float = 0.5,
        beta: float = 0.22,
        max_steps: int = 30,
        lr: float = 0.001,
        use_spectral_norm: bool = True,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.beta = beta
        self.max_steps = max_steps
        self.lr = lr
        self.use_spectral_norm = use_spectral_norm
        
        # Xavier initialization
        scale = 0.5
        self.W_in = np.random.randn(hidden_dim, input_dim).astype(np.float32) * scale / np.sqrt(input_dim)
        self.W_rec = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale / np.sqrt(hidden_dim)
        self.W_out = np.random.randn(output_dim, hidden_dim).astype(np.float32) * scale / np.sqrt(hidden_dim)
        
        self.b_in = np.zeros(hidden_dim, dtype=np.float32)
        self.b_rec = np.zeros(hidden_dim, dtype=np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)
        
        # Spectral norm state
        self.u_rec = None
        
        # Stats
        self.stats = {'steps': [], 'losses': []}
    
    def _get_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get weights with spectral normalization if enabled."""
        W_in = self.W_in
        W_out = self.W_out
        
        if self.use_spectral_norm:
            W_rec, self.u_rec, _ = spectral_normalize(self.W_rec, self.u_rec)
        else:
            W_rec = self.W_rec
        
        return W_in, W_rec, W_out
    
    def forward_step(self, h: np.ndarray, x_proj: np.ndarray, W_rec: np.ndarray) -> np.ndarray:
        """Single equilibrium step."""
        pre_act = x_proj + h @ W_rec.T + self.b_rec
        h_new = np.tanh(pre_act)
        return (1 - self.gamma) * h + self.gamma * h_new
    
    def solve_equilibrium(self, x: np.ndarray, nudge_grad: np.ndarray = None) -> Tuple[np.ndarray, int]:
        """Find equilibrium state via fixed-point iteration."""
        W_in, W_rec, _ = self._get_weights()
        batch_size = x.shape[0]
        
        # Embed input
        x_proj = x @ W_in.T + self.b_in
        
        # Initialize hidden
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        
        epsilon = 1e-4
        for step in range(self.max_steps):
            h_prev = h.copy()
            h = self.forward_step(h, x_proj, W_rec)
            
            # Apply nudge if in nudged phase
            if nudge_grad is not None:
                h = h - self.beta * nudge_grad
            
            # Check convergence
            diff = np.max(np.abs(h - h_prev))
            if diff < epsilon:
                return h, step + 1
        
        return h, self.max_steps
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: return logits."""
        h, _ = self.solve_equilibrium(x)
        return h @ self.W_out.T + self.b_out
    
    def train_step(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Full EqProp training step with contrastive Hebbian learning.
        
        This is TRUE EqProp: gradients computed from equilibrium difference,
        not autograd. Memory is O(1) because no graph is stored.
        """
        batch_size = x.shape[0]
        W_in, W_rec, W_out = self._get_weights()
        
        # === Free Phase ===
        h_free, steps_free = self.solve_equilibrium(x)
        
        # Compute output and loss gradient for nudging
        logits = h_free @ W_out.T + self.b_out
        probs = softmax(logits)
        
        # One-hot encoding
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), y] = 1.0
        
        # Output gradient: ∂L/∂logits
        d_logits = probs - one_hot
        
        # Nudge gradient: project back to hidden space
        nudge_grad = d_logits @ W_out
        
        # === Nudged Phase ===
        h_nudged, steps_nudged = self.solve_equilibrium(x, nudge_grad)
        
        # === Contrastive Hebbian Updates ===
        # ΔW = (1/β) * (A_nudged - A_free)
        # where A = outer product of activations
        
        x_proj = x @ W_in.T + self.b_in
        
        # Update W_rec: local Hebbian correlation
        # ΔW_rec = (1/β) * (h_nudged.T @ h_nudged - h_free.T @ h_free) / batch
        dW_rec = (h_nudged.T @ h_nudged - h_free.T @ h_free) / (self.beta * batch_size)
        
        # Update W_in: input-hidden correlation change
        dW_in = (h_nudged.T @ x - h_free.T @ x) / (self.beta * batch_size)
        
        # Update W_out: supervised (standard gradient)
        dW_out = d_logits.T @ h_free / batch_size
        
        # Update biases
        db_rec = (h_nudged.mean(axis=0) - h_free.mean(axis=0)) / self.beta
        db_out = d_logits.mean(axis=0)
        
        # === Apply updates (SGD for simplicity) ===
        self.W_rec -= self.lr * dW_rec
        self.W_in -= self.lr * dW_in
        self.W_out -= self.lr * dW_out
        self.b_rec -= self.lr * db_rec
        self.b_out -= self.lr * db_out
        
        # === Metrics ===
        loss = cross_entropy(logits, y)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        
        return {
            'loss': loss,
            'accuracy': acc,
            'free_steps': steps_free,
            'nudged_steps': steps_nudged,
        }
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate accuracy on a batch."""
        logits = self.predict(x)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        loss = cross_entropy(logits, y)
        return {'accuracy': acc, 'loss': loss}
    
    def get_memory_usage(self) -> Dict:
        """Estimate memory usage (bytes)."""
        # Parameters
        param_bytes = (
            self.W_in.nbytes + self.W_rec.nbytes + self.W_out.nbytes +
            self.b_in.nbytes + self.b_rec.nbytes + self.b_out.nbytes
        )
        
        # Activations (O(1) - only current state)
        # Batch of 32, hidden_dim floats
        activation_bytes = 32 * self.hidden_dim * 4  # 32-bit floats
        
        return {
            'parameters_bytes': param_bytes,
            'activations_bytes': activation_bytes,
            'total_bytes': param_bytes + activation_bytes,
            'total_mb': (param_bytes + activation_bytes) / 1e6,
        }


def compare_memory_autograd_vs_kernel(hidden_dim: int, depth: int) -> Dict:
    """
    Compare memory: PyTorch autograd vs NumPy kernel.
    
    PyTorch autograd: O(depth) - stores computational graph
    NumPy kernel: O(1) - only stores current state
    """
    # Kernel memory (O(1))
    kernel_activation = 32 * hidden_dim * 4  # Only current state
    
    # Autograd memory (O(depth))
    autograd_activation = 32 * hidden_dim * depth * 4  # All intermediate states
    
    return {
        'kernel_activation_mb': kernel_activation / 1e6,
        'autograd_activation_mb': autograd_activation / 1e6,
        'ratio': autograd_activation / kernel_activation,
    }
