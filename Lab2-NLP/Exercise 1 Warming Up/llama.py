from typing import Optional, Tuple
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer as T
import math
from einops import rearrange, repeat

"""
time to build a llama like Transformer model:
    - 1. tokenizer v
    - 2. positional encoding + embeddings
    - 4. multi head attention
    - 5. MLP
    - 5. encoder
"""


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_len=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1")
        self.max_len = max_len

    def forward(self, string):
        inputs = self.tokenizer(string, return_tensors="pt", max_length=self.max_len, truncation=True)
        token_id = inputs.input_ids
        # attention_mask = inputs.attention_mask
        return self.embedding(token_id)


class RMSNorm(torch.nn.Module):
    def __init__(self, emb_dimension: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dimension))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def freq_pos_enc(dim, max_seq_len):
    # inverse frequencies [theta_1, theta_2, ..., theta_dim/2]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))  # (dim/2,)
    # positions up to max_seq_len
    pos = torch.arange(max_seq_len)  # (max_seq_len,)
    # frequency position encoding (outer product: pos x inv_freq)
    pos_enc = torch.einsum("p, f -> p f", pos, inv_freq)  # (max_seq_len, dim/2)
    # duplicate each element along inverse frequency dimension
    return repeat(pos_enc, "... f -> ... (f r)", r=2)  # (max_seq_len, dim)


def rotate_half(u):
    # last dimension of u is [u1, u2, u3, u4, ...]
    u = rearrange(u, '... (d r) -> ... d r', r=2)
    u1, u2 = u.unbind(dim=-1)
    u = torch.stack((-u2, u1), dim=-1)
    # last dimension of result is [-u2, u1, -u4, u3, ...]
    return rearrange(u, '... d r -> ... (d r)')


def rotate(u, pos_enc):
    num_tokens = u.shape[-2]
    pos_enc = pos_enc[:num_tokens]
    return u * pos_enc.cos() + (rotate_half(u) * pos_enc.sin())


class MultiHeadAttention(nn.Module):
    # TODO: I'm not secure is this all correct
    def __init__(self, embed_dim, num_heads=8, max_seq_len=4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.freq_pos_enc = freq_pos_enc(self.d_k, max_seq_len)

        self.k_cache = None
        self.v_cache = None

        self.wq = nn.Linear(embed_dim, self.d_k * num_heads, bias=False)
        self.wk = nn.Linear(embed_dim, self.d_k * num_heads, bias=False)
        self.wv = nn.Linear(embed_dim, self.d_k * num_heads, bias=False)
        self.wo = nn.Linear(self.d_k * num_heads, embed_dim, bias=False)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape  # bsz, seq_len, embed_dim
        q, k, v = self.wq(x), self.wk(x), self.wv(x)  # bsz, seq_len, d_k * num_heads [1, 2, 4096]
        q = q.view(bsz,  seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [1, 8, 2, 512]
        k = k.view(bsz,  seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(bsz,  seq_len, self.num_heads, self.d_k).transpose(1, 2)

        q, k = rotate(q, self.freq_pos_enc), rotate(k, self.freq_pos_enc)

        if self.k_cache:
            old_k, old_v = self.k_cache, self.v_cache
            k = torch.cat(old_k, k)
            v = torch.cat(old_v, v)
        else:
            self.k_cache, self.v_cache = k, v

        mask = T.generate_square_subsequent_mask(k.shape[2])
        attention_score = (torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)) + mask  # [1, 8, 2, 2]

        attention_weights = F.softmax(attention_score, dim=-1)
        attention = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1) #[1, 2, 4096]
        return self.wo(attention)


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension=4096, intermediate_size=11008):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dimension, intermediate_size, bias=False)
        self.linear3 = nn.Linear(intermediate_size, embedding_dimension, bias=False)
        self.linear2 = nn.Linear(embedding_dimension, intermediate_size, bias=False)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = RMSNorm(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x, mask=None):
        out = x + self.attention(self.norm1(x), mask)
        out += self.ff(self.norm2(x))
        return out


class LittleLLama(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads=8, num_layers=2, max_seq_len=1024):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embedding_size, padding_idx=-1)
        self.layers = nn.ModuleList([TransformerLayer(embedding_size, num_heads=num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(embedding_size)

        # self.transformer = nn.Sequential(*[TransformerBlock(embedding_size, num_heads) for _ in range(num_layers)])
        # self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x)
        #x = self.transformer(x)
        #x = self.norm(x)
        return self.linear(x)


if __name__ == "__main__":
    llama = LittleLLama(32000, 4096)  # 32000 is the vocab size, 4096 is the embedding size
    # number of parameters:
    pytorch_total_params = sum(p.numel() for p in llama.parameters())
    print(pytorch_total_params)
    # llama('hello world')

    pass
