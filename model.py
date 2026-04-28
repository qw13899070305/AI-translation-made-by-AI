import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

cfg = Config()
cfg.validate()

# =========================== 基础组件 ===========================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim, end, theta=10000.0, scaling_factor=1.0):
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    t = torch.arange(end, dtype=torch.float32) / scaling_factor
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_ntk(dim, end, theta=10000.0, scaling_factor=1.0,
                              original_max_len=512, target_max_len=None):
    if target_max_len is None:
        target_max_len = end
    scale = target_max_len / max(original_max_len, 1)
    ntk_theta = theta * (scale ** (dim / (dim - 2)))
    return precompute_freqs_cis(dim, end, theta=ntk_theta, scaling_factor=scaling_factor)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[-2]].to(xq_.device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =========================== 注意力模块 ===========================

class GroupedQueryAttention(nn.Module):
    """GQA（分组查询注意力）"""
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.max_seq_len = max_seq_len
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        if cfg.use_ntk_rope:
            self.register_buffer("freqs_cis", precompute_freqs_cis_ntk(
                self.head_dim, max_seq_len * 2, cfg.rope_theta, cfg.rope_scaling_factor,
                cfg.original_max_seq_len, cfg.target_context_len))
        else:
            self.register_buffer("freqs_cis", precompute_freqs_cis(
                self.head_dim, max_seq_len * 2, cfg.rope_theta, cfg.rope_scaling_factor))

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        q, k = apply_rotary_emb(q, k, self.freqs_cis)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present_kv = (k, v) if use_cache else None

        if use_cache and k.shape[2] > self.max_seq_len:
            k, v = k[:,:,-self.max_seq_len:,:], v[:,:,-self.max_seq_len:,:]
            present_kv = (k, v)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        att_mask = mask[:, :, :T, :k.shape[2]] if mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=att_mask)
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.wo(out), present_kv


class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力 - 参考 MiMo-V2-Flash"""
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len, window_size=None):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.window_size = window_size or cfg.swa_window_size
        self.max_seq_len = max_seq_len
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        if cfg.use_ntk_rope:
            self.register_buffer("freqs_cis", precompute_freqs_cis_ntk(
                self.head_dim, max_seq_len * 2, cfg.rope_theta, cfg.rope_scaling_factor,
                cfg.original_max_seq_len, cfg.target_context_len))
        else:
            self.register_buffer("freqs_cis", precompute_freqs_cis(
                self.head_dim, max_seq_len * 2, cfg.rope_theta, cfg.rope_scaling_factor))
        self.register_buffer("attn_sink_bias", torch.zeros(1, n_heads, 1, 1))

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        q, k = apply_rotary_emb(q, k, self.freqs_cis)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        T_kv = k.shape[2]
        present_kv = (k, v) if use_cache else None

        if use_cache and T_kv > self.max_seq_len:
            k, v = k[:,:,-self.max_seq_len:,:], v[:,:,-self.max_seq_len:,:]
            T_kv = k.shape[2]
            present_kv = (k, v)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        att = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5) + self.attn_sink_bias

        arange = torch.arange(T_kv, device=x.device)
        q_pos = arange[-T:] if T <= T_kv else arange[:T]
        dist = (q_pos.unsqueeze(1) - arange.unsqueeze(0)).abs()
        swa_mask = dist <= (self.window_size // 2)
        att = att.masked_fill(~swa_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if mask is not None:
            att = att.masked_fill(mask[:,:,:T,:k.shape[2]] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = torch.where(torch.isnan(att), torch.ones_like(att)/att.shape[-1], att)
        out = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.wo(out), present_kv


class MultiHeadLatentAttention(nn.Module):
    """MLA - 参考 DeepSeek-V3"""
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.q_lora_rank = cfg.mla_q_lora_rank
        self.kv_lora_rank = cfg.mla_kv_lora_rank
        self.qk_rope_head_dim = cfg.mla_qk_rope_head_dim
        self.v_head_dim = cfg.mla_v_head_dim
        self.max_seq_len = max_seq_len

        self.q_a_proj = nn.Linear(dim, self.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, n_heads * self.qk_rope_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, n_heads * (self.qk_rope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(n_heads * self.v_head_dim, dim, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            self.qk_rope_head_dim, max_seq_len * 2, cfg.rope_theta, cfg.rope_scaling_factor))

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        B, T, C = x.shape
        q_compressed = self.q_a_norm(self.q_a_proj(x))
        q = self.q_b_proj(q_compressed).view(B, T, self.n_heads, self.qk_rope_head_dim)

        kv_a = self.kv_a_proj_with_mqa(x)
        kv_compressed, k_rope_raw = torch.split(kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_compressed = self.kv_a_norm(kv_compressed)

        if past_kv is not None:
            kv_compressed = torch.cat([past_kv[0], kv_compressed], dim=1)
            k_rope_raw = torch.cat([past_kv[1], k_rope_raw], dim=1)
        T_total = kv_compressed.shape[1]
        present_kv = (kv_compressed, k_rope_raw) if use_cache else None

        kv = self.kv_b_proj(kv_compressed).view(B, T_total, self.n_heads, self.qk_rope_head_dim + self.v_head_dim)
        k_content, v = torch.split(kv, [self.qk_rope_head_dim, self.v_head_dim], dim=-1)
        k_rope = k_rope_raw.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        q, k_rope = apply_rotary_emb(q, k_rope, self.freqs_cis)
        k = torch.cat([k_content, k_rope], dim=-1)

        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (self.qk_rope_head_dim ** -0.5)
        if mask is not None:
            att = att.masked_fill(mask[:,:,:T,:k.shape[2]] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = torch.where(torch.isnan(att), torch.ones_like(att)/att.shape[-1], att)
        out = (att @ v).transpose(1,2).contiguous().view(B, T_total, self.n_heads * self.v_head_dim)
        return self.o_proj(out[:, -T:]), present_kv


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet 线性注意力 (参考 Qwen3-Next)"""
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.gate_alpha = nn.Sequential(nn.Linear(dim, n_heads * self.head_dim), nn.Sigmoid())
        self.gate_beta  = nn.Sequential(nn.Linear(dim, n_heads * self.head_dim), nn.Sigmoid())
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        B, T, C = x.shape
        q = self.norm_q(self.wq(x).view(B, T, self.n_heads, self.head_dim))
        k = self.norm_k(self.wk(x).view(B, T, self.n_heads, self.head_dim))
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        alpha = self.gate_alpha(x).view(B, T, self.n_heads, self.head_dim)
        beta  = self.gate_beta(x).view(B, T, self.n_heads, self.head_dim)

        if use_cache and past_kv is not None:
            state = past_kv[0]
        else:
            state = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim,
                                device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            qt = q[:, t, :, :]
            kt = k[:, t, :, :]
            vt = v[:, t, :, :]
            at = alpha[:, t, :, :]
            bt = beta[:, t, :, :]

            out_t = torch.einsum('bhd,bhde->bhe', qt * bt, state)
            value_pred = torch.einsum('bhd,bhde->bhe', qt, state)
            error = vt - value_pred * at
            state = state * at.unsqueeze(-1) + torch.einsum('bhd,bhe->bhde', kt, error)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).contiguous().view(B, T, self.n_heads * self.head_dim)
        present_kv = (state,) if use_cache else None
        return self.wo(out), present_kv


class CompressedSparseAttention(nn.Module):
    """CSA 压缩稀疏注意力 (参考 DeepSeek V4)"""
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.compress_ratio = cfg.csa_compress_ratio
        self.top_k = min(cfg.csa_top_k, max_seq_len // cfg.csa_compress_ratio)
        self.max_seq_len = max_seq_len

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.indexer_q = nn.Linear(dim, self.head_dim, bias=False)
        self.indexer_k = nn.Linear(dim, self.head_dim, bias=False)
        self.compress_weight_a = nn.Linear(self.head_dim, 1, bias=False)
        self.compress_weight_b = nn.Linear(self.head_dim, 1, bias=False)

        if cfg.use_ntk_rope:
            self.register_buffer("freqs_cis", precompute_freqs_cis_ntk(
                self.head_dim, max_seq_len * 2, cfg.rope_theta, cfg.rope_scaling_factor,
                cfg.original_max_seq_len, cfg.target_context_len))
        else:
            self.register_buffer("freqs_cis", precompute_freqs_cis(
                self.head_dim, max_seq_len * 2, cfg.rope_theta, cfg.rope_scaling_factor))

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        q, k = apply_rotary_emb(q, k, self.freqs_cis)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        T_kv = k.shape[2]

        compressed_k, compressed_v = [], []
        for i in range(0, T_kv, self.compress_ratio):
            chunk_k = k[:, :, i:i+self.compress_ratio, :]
            chunk_v = v[:, :, i:i+self.compress_ratio, :]
            if chunk_k.shape[2] == 0: continue
            w_a = F.softmax(self.compress_weight_a(chunk_k), dim=2)
            w_b = F.softmax(self.compress_weight_b(chunk_k), dim=2)
            w = (w_a + w_b) / 2.0
            comp_k = (chunk_k * w).sum(dim=2)
            comp_v = (chunk_v * w).sum(dim=2)
            compressed_k.append(comp_k)
            compressed_v.append(comp_v)

        compressed_k = torch.stack(compressed_k, dim=2)
        compressed_v = torch.stack(compressed_v, dim=2)

        idxer_q = self.indexer_q(x).view(B, T, 1, self.head_dim).transpose(1,2)
        idxer_k = self.indexer_k(x).view(B, 1, T, self.head_dim)
        scores = (idxer_q @ idxer_k.transpose(-2,-1)).mean(dim=3)
        block_scores = scores[:, :, :, :compressed_k.shape[2]].squeeze(1)

        if compressed_k.shape[2] > self.top_k:
            ret = torch.topk(block_scores, self.top_k, dim=-1)
            topk_idx = ret.indices
            B_idx = torch.arange(B, device=x.device).view(B,1,1)
            H_idx = torch.arange(self.n_kv_heads, device=x.device).view(1,self.n_kv_heads,1)
            compressed_k = compressed_k[B_idx, H_idx, topk_idx.unsqueeze(1), :].permute(0,2,1,3).contiguous()
            compressed_v = compressed_v[B_idx, H_idx, topk_idx.unsqueeze(1), :].permute(0,2,1,3).contiguous()
            compressed_k = compressed_k.permute(0,2,1,3)
            compressed_v = compressed_v.permute(0,2,1,3)

        k_rep = compressed_k.repeat_interleave(self.n_rep, dim=1)
        v_rep = compressed_v.repeat_interleave(self.n_rep, dim=1)

        att = (q @ k_rep.transpose(-2,-1)) * (self.head_dim ** -0.5)
        att = F.softmax(att, dim=-1)
        out = (att @ v_rep).transpose(1,2).contiguous().view(B, T, C)
        present_kv = (compressed_k, compressed_v) if use_cache else None
        return self.wo(out), present_kv


# =========================== 创建注意力层 ===========================
def create_attention_layer(dim, n_heads, n_kv_heads, max_seq_len, layer_idx):
    attn_type = cfg.attn_type
    if attn_type == "gqa":
        return GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
    elif attn_type == "mla":
        return MultiHeadLatentAttention(dim, n_heads, max_seq_len)
    elif attn_type == "swa":
        return SlidingWindowAttention(dim, n_heads, n_kv_heads, max_seq_len)
    elif attn_type == "csa":
        return CompressedSparseAttention(dim, n_heads, n_kv_heads, max_seq_len)
    elif attn_type == "hybrid":
        ratio = cfg.swa_hybrid_ratio
        is_global = (layer_idx % ratio == ratio - 1)
        if is_global:
            return GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        else:
            return SlidingWindowAttention(dim, n_heads, n_kv_heads, max_seq_len)
    elif attn_type == "hybrid_gdn":
        ratio = cfg.gdn_hybrid_ratio
        if (layer_idx % ratio) == (ratio - 1):
            return GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        else:
            return GatedDeltaNet(dim, n_heads, max_seq_len)
    else:
        return GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)


# =========================== 增强 MoE ===========================
class QwenStyleMoE(nn.Module):
    """高稀疏度 MoE，带共享专家和动态偏置"""
    def __init__(self, dim, hidden_dim, num_experts=32, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)

        self.shared_expert = nn.Sequential(
            nn.Linear(dim, hidden_dim // 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, dim, bias=False)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False)
            ) for _ in range(num_experts)
        ])
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        shared_out = self.shared_expert(x_flat)

        scores = self.gate(x_flat).sigmoid() + self.e_score_correction_bias
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        sparse_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask_i = (topk_indices == i).any(dim=-1)
            if not mask_i.any():
                continue
            x_i = x_flat[mask_i]
            idx_in_topk = (topk_indices[mask_i] == i).nonzero(as_tuple=True)[1]
            weight_i = topk_weights[mask_i][torch.arange(x_i.shape[0], device=x.device), idx_in_topk]
            sparse_out[mask_i] += weight_i.unsqueeze(-1) * expert(x_i)

        return (shared_out + sparse_out).view(B, T, D)


# =========================== MTP 模块 ===========================
class MultiTokenPredictor(nn.Module):
    def __init__(self, dim, vocab_size, num_layers=None, hidden_dim=None):
        super().__init__()
        n_layers = num_layers or cfg.mtp_num_layers
        h_dim = hidden_dim or cfg.mtp_hidden_dim
        self.mtp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, h_dim, bias=False),
                nn.SiLU(),
                nn.Linear(h_dim, vocab_size, bias=False)
            ) for _ in range(n_layers)
        ])
        self.embed_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states):
        h = self.embed_norm(hidden_states)
        predictions = []
        for head in self.mtp_heads:
            pred = head(h)
            predictions.append(pred)
            h = F.linear(F.softmax(pred, dim=-1), self.mtp_heads[0][0].weight[:h.shape[-1]].t())
        return predictions


# =========================== Transformer Block ===========================
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len, use_moe=True, layer_idx=0):
        super().__init__()
        self.attn = create_attention_layer(dim, n_heads, n_kv_heads, max_seq_len, layer_idx)
        if use_moe:
            self.ffn = QwenStyleMoE(dim, dim * 4, num_experts=cfg.num_experts, top_k=cfg.top_k_experts)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4, bias=False),
                nn.SiLU(),
                nn.Linear(dim * 4, dim, bias=False)
            )
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        attn_out, present_kv = self.attn(self.norm1(x), mask, use_cache, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present_kv


# =========================== MiniChat 主模型 ===========================
class MiniChat(nn.Module):
    def __init__(self, vocab_size, dim=cfg.dim, n_layers=cfg.n_layers,
                 n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                 max_seq_len=cfg.max_seq_len, use_moe=cfg.use_moe):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, max_seq_len, use_moe, layer_idx=i)
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1,1,max_seq_len,max_seq_len)
        self.register_buffer("mask", mask)

        self.mtp = None
        if cfg.use_mtp:
            self.mtp = MultiTokenPredictor(dim, vocab_size)

    def forward(self, idx, targets=None, use_cache=False, past_kvs=None, vision_embeds=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        if vision_embeds is not None:
            x = torch.cat([vision_embeds, x], dim=1)
            T = x.shape[1]

        present_kvs = [] if use_cache else None
        last_hidden = x
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            last_hidden, present_kv = block(last_hidden, self.mask, use_cache, past_kv)
            if use_cache:
                present_kvs.append(present_kv)

        last_hidden = self.norm(last_hidden)
        logits = self.lm_head(last_hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                                   targets[:, 1:].contiguous().view(-1))

        mtp_predictions = self.mtp(last_hidden) if self.mtp else None
        return logits, loss, present_kvs, mtp_predictions

    def generate(self, idx, max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k,
                 vision_embeds=None, use_mtp_spec=False):
        if use_mtp_spec and self.mtp:
            return self._generate_mtp(idx, max_new_tokens, temperature, top_k, vision_embeds)
        return self._generate_vanilla(idx, max_new_tokens, temperature, top_k, vision_embeds)

    def _generate_vanilla(self, idx, max_new_tokens, temperature, top_k, vision_embeds):
        past_kvs = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _, past_kvs, _ = self.forward(
                idx_cond, use_cache=True, past_kvs=past_kvs,
                vision_embeds=vision_embeds if past_kvs is None else None)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == self.token_embedding.num_embeddings - 1:
                break
        return idx

    def _generate_mtp(self, idx, max_new_tokens, temperature, top_k, vision_embeds):
        past_kvs = None
        mtp_depth = len(self.mtp.mtp_heads)
        step = 0
        while step < max_new_tokens and idx.shape[1] < self.max_seq_len:
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _, past_kvs, mtp_preds = self.forward(
                idx_cond, use_cache=True, past_kvs=past_kvs,
                vision_embeds=vision_embeds if past_kvs is None else None)

            # 基准 token
            logits_base = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits_base, min(top_k, logits_base.size(-1)))
                logits_base[logits_base < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits_base, dim=-1)
            base_token = torch.multinomial(probs, num_samples=1)
            draft_tokens = [base_token]

            # MTP 预测
            for i in range(mtp_depth):
                mtp_logits = mtp_preds[i][:, -1, :] / temperature
                if top_k:
                    v, _ = torch.topk(mtp_logits, min(top_k, mtp_logits.size(-1)))
                    mtp_logits[mtp_logits < v[:, [-1]]] = -float('Inf')
                draft_tokens.append(torch.multinomial(F.softmax(mtp_logits, dim=-1), 1))

            # 并行验证
            draft_ids = torch.cat(draft_tokens, dim=1)
            verify_in = torch.cat([idx_cond, draft_ids], dim=1)
            with torch.no_grad():
                verify_logits, _, _, _ = self.forward(verify_in)
            accepted = [base_token]
            for i in range(len(draft_tokens)-1):
                pred_token = torch.argmax(verify_logits[:, idx_cond.shape[1]+i, :], dim=-1, keepdim=True)
                if pred_token.item() == draft_tokens[i+1].item():
                    accepted.append(pred_token)
                else:
                    accepted.append(pred_token)
                    break
            idx = torch.cat([idx, torch.cat(accepted, dim=1)], dim=1)
            step += len(accepted)
        return idx