import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.utils import linear_attention

class LoopedTransformerBlock(nn.Module):
    """Single weight-tied transformer block for equilibrium iteration."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, use_linear_attn: bool = False, symmetric: bool = False):
        super().__init__()
        self.use_linear_attn = use_linear_attn
        self.symmetric = symmetric
        self.d_model = d_model
        self.n_heads = n_heads

        # Standard transformer components
        if not use_linear_attn:
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        else:
            if symmetric:
                # Symmetric Linear Attention constraints:
                # W_out = W_q^T
                # W_k = W_v
                self.w_q = nn.Linear(d_model, d_model, bias=True)
                self.w_k = nn.Linear(d_model, d_model, bias=True)
            else:
                self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
                self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
                self.out_proj = nn.Linear(d_model, d_model)
                self._reset_parameters()

        if symmetric:
            # Symmetric FFN: W2 = W1.T
            self.w_ff1 = nn.Linear(d_model, d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )

        if not symmetric:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            # Add final norm to ensure stability of fixed point (Universal Transformer style)
            self.norm_final = nn.LayerNorm(d_model)
        else:
            # No norms in symmetric mode for now, rely on Tanh
            pass

    def _reset_parameters(self):
        if hasattr(self, 'in_proj_weight'):
            nn.init.xavier_uniform_(self.in_proj_weight)
            nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        # h: [seq, batch, d_model], x: [seq, batch, d_model]

        if not self.symmetric:
            h_in = h
            h_norm = self.norm1(h)

            if not self.use_linear_attn:
                attn_out, _ = self.attn(h_norm, x, x)
            else:
                w_q, w_k, w_v = self.in_proj_weight.chunk(3)
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)

                q = F.linear(h_norm, w_q, b_q).permute(1, 0, 2)
                k = F.linear(x, w_k, b_k).permute(1, 0, 2)
                v = F.linear(x, w_v, b_v).permute(1, 0, 2)

                B, S, D = q.shape
                head_dim = D // self.n_heads
                q = q.view(B, S, self.n_heads, head_dim)
                k = k.view(B, S, self.n_heads, head_dim)
                v = v.view(B, S, self.n_heads, head_dim)

                q = q.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)
                k = k.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)
                v = v.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)

                attn_out = linear_attention(q, k, v)

                attn_out = attn_out.view(B, self.n_heads, S, head_dim)
                attn_out = attn_out.permute(2, 0, 1, 3).reshape(S, B, D)

                attn_out = self.out_proj(attn_out)

            h = h + attn_out
            h_norm = self.norm2(h)
            h = h + self.ffn(h_norm)

            # Final normalization to ensure boundedness
            h = self.norm_final(h)
            return h

        else:
            # Symmetric Mode (Hopfield / Energy Model)
            # No norms, use Tanh

            # Linear Attn
            q = self.w_q(h).permute(1, 0, 2)
            k = self.w_k(x).permute(1, 0, 2)
            v = k

            B, S, D = q.shape
            head_dim = D // self.n_heads
            q = q.view(B, S, self.n_heads, head_dim)
            k = k.view(B, S, self.n_heads, head_dim)
            v = v.view(B, S, self.n_heads, head_dim)

            q = q.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)
            k = k.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)
            v = v.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, head_dim)

            attn_out = linear_attention(q, k, v)

            attn_out = attn_out.view(B, self.n_heads, S, head_dim)
            attn_out = attn_out.permute(2, 0, 1, 3).reshape(S, B, D)

            # W_out = W_q.T
            attn_out = F.linear(attn_out, self.w_q.weight.t())

            # FFN
            # FFN = W1.T @ Tanh(W1 @ h)
            h_ff = self.w_ff1(h)
            h_ff = torch.tanh(h_ff) # Use Tanh for bounded energy
            h_ff = F.linear(h_ff, self.w_ff1.weight.t())

            # Combine
            # h_new = Tanh(h + Attn + FFN)
            return torch.tanh(h + attn_out + h_ff)
