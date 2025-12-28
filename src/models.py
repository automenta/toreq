import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import linear_attention

class LoopedTransformerBlock(nn.Module):
    """Single weight-tied transformer block for equilibrium iteration.
    
    Supports both standard and symmetric modes:
    - Standard mode: Uses regular attention and FFN with LayerNorm
    - Symmetric mode: Implements weight tying (W_out=W_q^T, W_k=W_v, W2=W1^T)
      for energy-based dynamics required by EqProp theoretical guarantees.
      References: Scellier & Bengio 2017
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, 
                 attention_type: str = 'softmax', symmetric: bool = False):
        super().__init__()
        self.attention_type = attention_type
        self.symmetric = symmetric
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Symmetric mode requires linear attention
        if symmetric and attention_type == 'softmax':
            raise ValueError("Symmetric mode requires attention_type='linear'")

        if attention_type == 'softmax':
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        elif attention_type == 'linear':
            if symmetric:
                # Symmetric Linear Attention constraints:
                # W_out = W_q^T (applied dynamically, not stored)
                # W_k = W_v (key and value share weights)
                self.w_q = nn.Linear(d_model, d_model, bias=True)
                self.w_k = nn.Linear(d_model, d_model, bias=True)
                # Note: w_k is used for both K and V projections
            else:
                # Non-symmetric linear attention
                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)

        # FFN configuration
        if symmetric:
            # Symmetric FFN: W2 = W1^T (output = input weight transposed)
            self.w_ff1 = nn.Linear(d_model, d_ff)
            # No separate w_ff2 - use w_ff1.weight.t() dynamically
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )

        # LayerNorm configuration
        if not symmetric:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            # Add final norm for Universal Transformer style stability
            self.norm_final = nn.LayerNorm(d_model)
        # Symmetric mode: no LayerNorm, rely on tanh bounding

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Forward pass with equilibrium dynamics.
        
        Args:
            h: Hidden state [seq, batch, d_model]
            x: Input [seq, batch, d_model]
            
        Returns:
            Updated hidden state [seq, batch, d_model]
        """
        if not self.symmetric:
            # Standard (non-symmetric) mode
            h_norm = self.norm1(h)

            if self.attention_type == 'softmax':
                # Cross-attend to input: Query=h, Key=x, Value=x
                attn_out, _ = self.attn(h_norm, x, x, need_weights=False)
            else:
                # Linear attention using custom implementation
                seq_len, batch_size, d_model = h.shape
                Q = self.q_proj(h_norm).view(seq_len, batch_size, self.n_heads, self.head_dim)
                K = self.k_proj(x).view(seq_len, batch_size, self.n_heads, self.head_dim)
                V = self.v_proj(x).view(seq_len, batch_size, self.n_heads, self.head_dim)

                # Permute to [batch, heads, seq, dim]
                Q = Q.permute(1, 2, 0, 3)  # [B, H, S, D]
                K = K.permute(1, 2, 0, 3)
                V = V.permute(1, 2, 0, 3)

                # Feature map phi(x) = elu(x) + 1
                Q_prime = F.elu(Q) + 1.0
                K_prime = F.elu(K) + 1.0

                # Efficient attention: (Q @ K^T) @ V -> Q @ (K^T @ V)
                KV = torch.einsum('bhsd,bhsv->bhdv', K_prime, V)
                Z = torch.einsum('bhsd,bhd->bhs', Q_prime, K_prime.sum(dim=2)) + 1e-6
                out = torch.einsum('bhsd,bhdv->bhsv', Q_prime, KV)
                out = out / Z.unsqueeze(-1)

                # Reshape back to [seq, batch, d_model]
                out = out.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, d_model)
                attn_out = self.out_proj(out)

            h = h + attn_out
            h_norm = self.norm2(h)
            h = h + self.ffn(h_norm)
            
            # Final normalization for stability
            h = self.norm_final(h)
            return h

        else:
            # Symmetric mode (Hopfield / Energy model)
            # No LayerNorm, use tanh for bounded energy
            
            # Linear attention with symmetric constraints
            seq_len, batch_size, d_model = h.shape
            
            # Project Q and K (V = K by weight sharing)
            Q = self.w_q(h)  # [seq, batch, d_model]
            K = self.w_k(x)  # [seq, batch, d_model]
            V = K  # W_k = W_v constraint
            
            # Reshape for multi-head attention
            # Permute to [batch, seq, d_model] for linear_attention
            Q = Q.permute(1, 0, 2)  # [B, S, D]
            K = K.permute(1, 0, 2)
            V = V.permute(1, 0, 2)
            
            B, S, D = Q.shape
            head_dim = D // self.n_heads
            Q = Q.view(B, S, self.n_heads, head_dim)
            K = K.view(B, S, self.n_heads, head_dim)
            V = V.view(B, S, self.n_heads, head_dim)
            
            Q = Q.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)
            K = K.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)
            V = V.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)
            
            # Apply linear attention
            attn_out = linear_attention(Q, K, V)  # [B*H, S, head_dim]
            
            # Reshape back
            attn_out = attn_out.view(B, self.n_heads, S, head_dim)
            attn_out = attn_out.permute(2, 0, 1, 3).reshape(S, B, D)  # [S, B, D]
            
            # Apply W_out = W_q^T constraint
            attn_out = F.linear(attn_out, self.w_q.weight.t())
            
            # Symmetric FFN: W2 = W1^T
            h_ff = self.w_ff1(h)  # [seq, batch, d_ff]
            h_ff = torch.tanh(h_ff)  # Use tanh for bounded energy
            h_ff = F.linear(h_ff, self.w_ff1.weight.t())  # [seq, batch, d_model]
            
            # Combine with tanh activation for overall bounded energy
            return torch.tanh(h + attn_out + h_ff)
