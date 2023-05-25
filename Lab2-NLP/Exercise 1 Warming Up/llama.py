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
    - Rotary Embedding
    - Mask
"""


def freq_pos_enc(dim, max_seq_len):
    # inverse frequencies [theta_1, theta_2, ..., theta_dim/2]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))  # (dim/2,)
    # positions up to max_seq_len
    pos = torch.arange(max_seq_len)  # (max_seq_len,)
    # frequency position encoding (outer product: pos x inv_freq)
    pos_enc = torch.einsum("p, f -> p f", pos, inv_freq)  # (max_seq_len, dim/2)
    # duplicate each element along inverse frequency dimension
    return repeat(pos_enc, "... f -> ... (f r)", r=2)  # (max_seq_len, dim)


def rotatee_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


def tokenization(string, max_length):
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1")
    inputs = tokenizer(string, return_tensors="pt", max_length=max_length, truncation=True)
    return inputs.input_ids


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


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :].to(dtype=x.dtype)
            self.sin_cached = emb.sin()[None, None, :, :].to(dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
        )


class MultiHeadAttention(nn.Module):
    # TODO: I'm not secure is this all correct
    def __init__(self, embed_dim, num_heads, d_k):
        super().__init__()
        self.embed_dim = embed_dim

        # self.rotary_emb = RotaryEmbedding(self.head_dim)

        self.k_cache = None
        self.v_cache = None

        self.wq = nn.Linear(embed_dim, d_k * num_heads, bias=False)
        self.wk = nn.Linear(embed_dim, d_k * num_heads, bias=False)
        self.wv = nn.Linear(embed_dim, d_k * num_heads, bias=False)
        self.wo = nn.Linear(d_k * num_heads, embed_dim, bias=False)

    def forward(self, x, num_heads, d_k, freq_pos_enc, mask):
        bsz, seq_len, _ = x.shape  # bsz, seq_len, embed_dim
        q, k, v = self.wq(x), self.wk(x), self.wv(x)  # bsz, seq_len, d_k * num_heads [1, 2, 4096]
        q = q.view(bsz, seq_len, num_heads, d_k).transpose(1, 2)  # [1, 8, 2, 512]
        k = k.view(bsz, seq_len, num_heads, d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, num_heads, d_k).transpose(1, 2)

        # kv_seq_len = k.shape[-2]

        # cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, offset=offset)

        q, k = rotate(q, freq_pos_enc), rotate(k, freq_pos_enc)

        if self.k_cache:
            old_k, old_v = self.k_cache, self.v_cache
            k = torch.cat([old_k, k], dim=2)
            v = torch.cat([old_v, v], dim=2)
        else:
            self.k_cache, self.v_cache = k, v

        mask = T.generate_square_subsequent_mask(k.shape[2])
        attention_score = (torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)) + mask  # [1, 8, 2, 2]

        attention_weights = F.softmax(attention_score, dim=-1)
        attention = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(attention)


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()
        intermediate_size = 256 * ((int(8 * embedding_dimension / 3) + 256 - 1) // 256)  # numbers from llama github :(
        self.linear1 = nn.Linear(embedding_dimension, intermediate_size, bias=False)
        self.linear3 = nn.Linear(intermediate_size, embedding_dimension, bias=False)
        self.linear2 = nn.Linear(embedding_dimension, intermediate_size, bias=False)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_k):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, d_k)
        self.norm1 = RMSNorm(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x, num_heads, d_k, freq_pos_enc, mask=None):
        out = x + self.attention(self.norm1(x), num_heads, d_k, freq_pos_enc, mask)
        out = out + self.ff(self.norm2(out))
        return out


class LittleLLama(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.d_k = embedding_size // num_heads
        self.freq_pos_enc = freq_pos_enc(self.d_k, max_seq_len)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=-1)

        self.norm = RMSNorm(embedding_size)
        self.layers = nn.Sequential(*[TransformerLayer(embedding_size, num_heads, self.d_k) for _ in range(num_layers)])
        self.out = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, x):
        x = tokenization(x, self.max_seq_len)
        x = self.embedding(x)
        x = self.layers(x, self.num_heads, self.d_k, self.freq_pos_enc, mask=None)
        x = self.norm(x)
        return self.out(x[:, -1, :]).float()  # only compute last logits


if __name__ == "__main__":
    llama = LittleLLama(vocab_size=1500,  # fa molto nella dimensione finale del modello
                        embedding_size=1024,  # fa abbastanza nella dimensione finale del modello
                        num_heads=8,
                        num_layers=4,
                        max_seq_len=1024)
    # number of parameters:
    pytorch_total_params = sum(p.numel() for p in llama.parameters())
    print(pytorch_total_params)
    # llama('hello world')
    pass
