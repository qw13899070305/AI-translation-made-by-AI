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
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            self.head_dim, max_seq_len * 2, theta=cfg.rope_theta, scaling_factor=cfg.rope_scaling_factor))

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

        att = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5)
        if mask is not None:
            att = att.masked_fill(mask[...,:T,:k.shape[2]] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        if torch.isnan(att).any():
            att = torch.ones_like(att) / att.shape[-1]
        out = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.wo(out), present_kv


class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力 - 参考 MiMo-V2-Flash: 128-token window, 5:1 hybrid ratio"""
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
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            self.head_dim, max_seq_len * 2, theta=cfg.rope_theta, scaling_factor=cfg.rope_scaling_factor))
        # Attention sink bias: 可学习偏置缓解 SWA 长距离衰减
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

        # 构造 SWA 掩码: 每个 token 只能看到前后 window_size/2 范围内的 token
        arange = torch.arange(T_kv, device=x.device)
        q_pos = arange[-T:] if T <= T_kv else arange[:T]
        dist = (q_pos.unsqueeze(1) - arange.unsqueeze(0)).abs()
        swa_mask = dist <= (self.window_size // 2)
        att = att.masked_fill(~swa_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # 叠加因果掩码
        if mask is not None:
            att = att.masked_fill(mask[...,:T,:k.shape[2]] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        if torch.isnan(att).any():
            att = torch.ones_like(att) / att.shape[-1]
        out = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.wo(out), present_kv


class MultiHeadLatentAttention(nn.Module):
    """MLA - 参考 DeepSeek-V3: KV低维压缩, Q压缩解压, 解耦RoPE"""
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
            self.qk_rope_head_dim, max_seq_len * 2, theta=cfg.rope_theta, scaling_factor=cfg.rope_scaling_factor))

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
            att = att.masked_fill(mask[...,:T,:k.shape[2]] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        if torch.isnan(att).any():
            att = torch.ones_like(att) / att.shape[-1]
        out = (att @ v).transpose(1,2).contiguous().view(B, T_total, self.n_heads * self.v_head_dim)
        return self.o_proj(out[:, -T:]), present_kv

# =========================== 混合注意力层 ===========================

def create_attention_layer(dim, n_heads, n_kv_heads, max_seq_len, layer_idx):
    """
    根据 attn_type 和 layer_idx 创建对应的注意力层。
    hybrid 模式: 每 (swa_hybrid_ratio) 层中，前 (swa_hybrid_ratio-1) 层为 SWA，最后 1 层为 GQA。
    """
    attn_type = cfg.attn_type
    if attn_type == "gqa":
        return GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
    elif attn_type == "mla":
        return MultiHeadLatentAttention(dim, n_heads, max_seq_len)
    elif attn_type == "swa":
        return SlidingWindowAttention(dim, n_heads, n_kv_heads, max_seq_len)
    elif attn_type == "hybrid":
        ratio = cfg.swa_hybrid_ratio  # 默认 5
        is_global = (layer_idx % ratio == ratio - 1)
        if is_global:
            return GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        else:
            return SlidingWindowAttention(dim, n_heads, n_kv_heads, max_seq_len)
    else:
        return GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)

# =========================== 增强 MoE ===========================

class MoEFeedForward(nn.Module):
    """
    增强 MoE: Sigmoid门控 + 分组限制 + 共享专家 + 动态偏置负载均衡
    参考 DeepSeek-V3 和 MiMo-V2-Flash
    """
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts, self.top_k, self.dim = num_experts, top_k, dim
        self.use_sigmoid = cfg.moe_use_sigmoid_gate
        self.num_shared = cfg.moe_num_shared_experts
        self.n_groups = cfg.moe_n_groups
        self.topk_group = cfg.moe_topk_group
        self.norm_topk = cfg.moe_norm_topk_prob
        self.routed_scale = cfg.moe_routed_scaling_factor

        self.gate = nn.Linear(dim, num_experts, bias=False)
        if cfg.moe_use_aux_loss_free:
            self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))
        else:
            self.e_score_correction_bias = None

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim, bias=False), nn.SiLU(), nn.Linear(hidden_dim, dim, bias=False))
            for _ in range(num_experts)
        ])
        self.shared_experts = None
        if self.num_shared > 0:
            self.shared_experts = nn.ModuleList([
                nn.Sequential(nn.Linear(dim, hidden_dim, bias=False), nn.SiLU(), nn.Linear(hidden_dim, dim, bias=False))
                for _ in range(self.num_shared)
            ])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        shared_out = 0
        if self.shared_experts is not None:
            for s_expert in self.shared_experts:
                shared_out = shared_out + s_expert(x_flat)

        gate_logits = self.gate(x_flat)
        scores = gate_logits.sigmoid() if self.use_sigmoid else F.softmax(gate_logits, dim=-1)
        if self.e_score_correction_bias is not None:
            scores = scores + self.e_score_correction_bias

        if self.n_groups > 1:
            group_size = self.num_experts // self.n_groups
            group_scores = scores.view(-1, self.n_groups, group_size)
            group_topk = group_scores.topk(min(2, group_size), dim=-1)[0].sum(dim=-1)
            group_idx = group_topk.topk(self.topk_group, dim=-1)[1]
            group_mask = F.one_hot(group_idx, self.n_groups).max(dim=1)[0].unsqueeze(-1)
            scores = (scores.view(-1, self.n_groups, group_size) * group_mask).view(-1, self.num_experts)

        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        if self.norm_topk:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scale

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask_i = (topk_indices == i).any(dim=-1)
            if not mask_i.any():
                continue
            x_i = x_flat[mask_i]
            if x_i.shape[0] == 0:
                continue
            expert_out = expert(x_i)
            idx_in_topk = (topk_indices[mask_i] == i).nonzero(as_tuple=True)[1]
            weight_i = topk_weights[mask_i][torch.arange(x_i.shape[0], device=x.device), idx_in_topk]
            out[mask_i] = out[mask_i] + weight_i.unsqueeze(-1) * expert_out

        return (shared_out + out).view(B, T, D)

# =========================== MTP 模块 ===========================

class MultiTokenPredictor(nn.Module):
    """MTP 多 Token 预测 - 参考 MiMo-V2-Flash: 3层轻量级头，自投机解码加速2.6x"""
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
        """返回 list of [B, T, vocab_size]"""
        h = self.embed_norm(hidden_states)
        predictions = []
        for head in self.mtp_heads:
            pred = head(h)
            predictions.append(pred)
            # 用预测结果的 softmax 嵌入作为下一层输入
            h = F.linear(F.softmax(pred, dim=-1), self.mtp_heads[0][0].weight[:h.shape[-1]].t())
        return predictions

# =========================== Transformer Block ===========================

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len, use_moe=False, layer_idx=0):
        super().__init__()
        self.attn = create_attention_layer(dim, n_heads, n_kv_heads, max_seq_len, layer_idx)
        if use_moe:
            self.ffn = MoEFeedForward(dim, dim*4, num_experts=cfg.num_experts, top_k=cfg.top_k_experts)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim*4, bias=False), nn.SiLU(), nn.Linear(dim*4, dim, bias=False))
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)

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
        # MTP
        self.mtp = None
        if cfg.use_mtp:
            self.mtp = MultiTokenPredictor(dim, vocab_size)

    def forward(self, idx, targets=None, use_cache=False, past_kvs=None, vision_embeds=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        if vision_embeds is not None:
            assert vision_embeds.shape[-1] == x.shape[-1] and vision_embeds.shape[0] == x.shape[0]
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

        # MTP 输出
        mtp_predictions = None
        if self.mtp is not None:
            mtp_predictions = self.mtp(last_hidden)

        return logits, loss, present_kvs, mtp_predictions

    def generate(self, idx, max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k, vision_embeds=None):
        past_kvs = None
        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _, past_kvs, _ = self.forward(
                idx_cond, use_cache=True, past_kvs=past_kvs,
                vision_embeds=vision_embeds if step == 0 else None)
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