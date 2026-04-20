import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

cfg = Config()

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0, scaling_factor=1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end) / scaling_factor
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]].to(xq_.device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_seq_len * 2, theta=cfg.rope_theta, scaling_factor=cfg.rope_scaling_factor))

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        q, k = apply_rotary_emb(q, k, self.freqs_cis)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v) if use_cache else None
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        att = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5)
        if mask is not None:
            att = att.masked_fill(mask[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.wo(out), present_kv

class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim, bias=False), nn.SiLU(), nn.Linear(hidden_dim, dim, bias=False))
            for _ in range(num_experts)
        ])
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        gate_logits = self.gate(x_flat)
        weights = F.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_flat[mask])
                weight = topk_weights[mask][topk_indices[mask] == i].unsqueeze(-1)
                out[mask] += weight * expert_out
        return out.view(B, T, D)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len, use_moe=False):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        if use_moe:
            self.ffn = MoEFeedForward(dim, dim*4, num_experts=cfg.num_experts, top_k=cfg.top_k_experts)
        else:
            self.ffn = nn.Sequential(nn.Linear(dim, dim*4, bias=False), nn.SiLU(), nn.Linear(dim*4, dim, bias=False))
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        attn_out, present_kv = self.attn(self.norm1(x), mask, use_cache, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present_kv

class MiniChat(nn.Module):
    def __init__(self, vocab_size, dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, max_seq_len=cfg.max_seq_len, use_moe=cfg.use_moe):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, n_kv_heads, max_seq_len, use_moe) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1,1,max_seq_len,max_seq_len)
        self.register_buffer("mask", mask)

    def forward(self, idx, targets=None, use_cache=False, past_kvs=None, vision_embeds=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        if vision_embeds is not None:
            x = torch.cat([vision_embeds, x], dim=1)
        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = block(x, self.mask, use_cache, past_kv)
            if use_cache:
                present_kvs.append(present_kv)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
        return logits, loss, present_kvs

    def generate(self, idx, max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k, vision_embeds=None):
        past_kvs = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _, past_kvs = self.forward(idx_cond, use_cache=True, past_kvs=past_kvs, vision_embeds=vision_embeds if _ == 0 else None)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == self.token_embedding.num_embeddings - 1:
                break
        return idx