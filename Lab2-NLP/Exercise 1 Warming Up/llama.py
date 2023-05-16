from typing import Optional, Tuple
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

"""
time to build a llama like Transformer model:
    - 1. tokenizer v
    - 2. positional encoding + embeddings
    - 4. multi head attention
    - 5. MLP
    - 5. encoder
"""


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, ) -> Tuple[
    torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MultiHeadAttention(nn.Module):
    # TODO: implement rotary embeddings
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.wq = nn.Linear(embed_dim, self.d_k * num_heads)
        self.wk = nn.Linear(embed_dim, self.d_k * num_heads)
        self.wv = nn.Linear(embed_dim, self.d_k * num_heads)
        self.wo = nn.Linear(self.d_k * num_heads, embed_dim)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape  # bsz, seq_len, embed_dim
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seq_len, self.num_heads, self.d_k)
        k = k.view(bsz, seq_len, self.num_heads, self.d_k)
        v = v.view(bsz, seq_len, self.num_heads, self.d_k)

        attention_score = torch.matmul(q, k) / math.sqrt(self.d_k)
        if mask is not None:
            attention_score = attention_score + mask
        attention_weights = F.softmax(attention_score, dim=-1)
        attention = torch.matmul(attention_weights, v)
        return self.wo(attention)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.seq(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = RMSNorm(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x):
        attention = self.attention(self.norm1(x))
        x = x + attention
        attention = self.ff(self.norm2(x))
        return x + attention


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1")
    inputs = tokenizer("Hello world", return_tensors="pt", max_length=4096)
    token_id = inputs.input_ids
    attention_mask = inputs.attention_mask
    print(token_id, attention_mask)
    attention = MultiHeadAttention(4096, 8)
    embed = nn.Embedding(32000, 4096)  # 32000 is the vocab size, 4096 is the embedding size
    out = attention(embed(token_id), attention_mask.float())
    print(out.shape)
    pass
