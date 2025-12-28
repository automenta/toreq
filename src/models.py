import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class LoopedTransformerBlock(nn.Module):
    """Single weight-tied transformer block for equilibrium iteration."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, attention_type: str = 'softmax'):
        super().__init__()
        self.attention_type = attention_type

        if attention_type == 'softmax':
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        elif attention_type == 'linear':
            # Implement linear attention manually or use a simplified version
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        # h: [seq, batch, d_model], x: [seq, batch, d_model]
        h_norm = self.norm1(h)

        if self.attention_type == 'softmax':
            # Cross-attend to input: Query=h, Key=x, Value=x
            attn_out, _ = self.attn(h_norm, x, x, need_weights=False)
        else:
            # Linear attention: phi(Q) @ (phi(K)^T @ V)
            # Shapes: [seq, batch, d_model]

            # Project
            seq_len, batch_size, d_model = h.shape
            Q = self.q_proj(h_norm).view(seq_len, batch_size, self.n_heads, self.head_dim)
            K = self.k_proj(x).view(seq_len, batch_size, self.n_heads, self.head_dim)
            V = self.v_proj(x).view(seq_len, batch_size, self.n_heads, self.head_dim)

            # Permute to [batch, heads, seq, dim]
            Q = Q.permute(1, 2, 0, 3) # [B, H, S, D]
            K = K.permute(1, 2, 0, 3)
            V = V.permute(1, 2, 0, 3)

            # Feature map phi(x) = elu(x) + 1
            Q_prime = F.elu(Q) + 1.0
            K_prime = F.elu(K) + 1.0

            # Efficient attention: (Q @ K^T) @ V -> Q @ (K^T @ V)
            # K_prime: [B, H, S, D]
            # V: [B, H, S, D]
            # KV = K^T @ V -> [B, H, D, D]

            KV = torch.einsum('bhsd,bhsv->bhdv', K_prime, V)

            # Z = Q @ K^T @ 1 = Q @ sum(K)
            # Q_prime: [B, H, S, D]
            # K_prime.sum(dim=2): [B, H, D]
            Z = torch.einsum('bhsd,bhd->bhs', Q_prime, K_prime.sum(dim=2)) + 1e-6

            # Out = Q @ KV
            # Q_prime: [B, H, S, D]
            # KV: [B, H, D, D]
            out = torch.einsum('bhsd,bhdv->bhsv', Q_prime, KV)

            # Normalize
            out = out / Z.unsqueeze(-1)

            # Reshape back
            out = out.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, d_model)
            attn_out = self.out_proj(out)

        h = h + attn_out
        h_norm = self.norm2(h)
        h = h + self.ffn(h_norm)
        return h
