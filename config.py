import torch

class Config:
    # ========== 数据 ==========
    text_datasets = [
        "Open-Orca/OpenOrca",
        "my_local_data.txt",
        "distillation.txt",
        "enhanced_data.txt"
    ]
    max_samples_per_dataset = 50000
    max_seq_len = 512
    multimodal_dataset = "liuhaotian/LLaVA-Instruct-150K"
    use_multimodal = True
    max_multimodal_samples = 1000

    # ========== 分词器 ==========
    vocab_size = 4096
    tokenizer_prefix = "our_bpe"

    # ========== 模型架构 ==========
    dim = 384
    n_layers = 8
    n_heads = 8
    n_kv_heads = 4
    use_moe = True
    num_experts = 64
    top_k_experts = 2

    moe_use_sigmoid_gate = True
    moe_num_shared_experts = 1
    moe_n_groups = 4
    moe_topk_group = 2
    moe_norm_topk_prob = True
    moe_routed_scaling_factor = 1.0
    moe_use_aux_loss_free = True

    rope_theta = 10000.0
    rope_scaling_factor = 1.0
    use_ntk_rope = True
    original_max_seq_len = 512
    target_context_len = 2048

    attn_type = "hybrid_gdn"
    swa_window_size = 128
    swa_hybrid_ratio = 5
    gdn_hybrid_ratio = 4

    mla_q_lora_rank = 192
    mla_kv_lora_rank = 128
    mla_qk_rope_head_dim = 32
    mla_v_head_dim = 64

    use_partial_rope = True
    partial_rope_dim = 64

    csa_compress_ratio = 4
    csa_top_k = 512

    use_mtp = True
    mtp_num_layers = 3
    mtp_hidden_dim = 512
    use_mtp_speculative_decode = True

    mixed_reasoning = True

    # ========== 多模态 ==========
    vision_encoder_name = "openai/clip-vit-base-patch32"
    vision_embed_dim = 768
    proj_dim = 384
    num_queries = 32
    qformer_layers = 2
    qformer_heads = 8

    # ========== LoRA ==========
    use_lora = True
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["wq", "wv", "kv_compress", "k_proj", "v_proj"]

    # ========== 训练 ==========
    batch_size = 16
    learning_rate = 1e-4
    epochs = 20
    grad_clip = 1.0
    save_every = 1
    num_workers = 16
    checkpoint_dir = "./checkpoints"
    lora_checkpoint_dir = "./lora_weights"
    use_amp = True

    # ========== 优化器 ==========
    use_muon = False
    use_muon_clip = False
    muon_clip_grad = 1.0
    muon_clip_update = 1.0

    # ========== 训练增强 ==========
    use_fp8 = False
    use_hta = False
    hta_warmup_steps = 100

    # ========== RAG ==========
    embedding_model = "BAAI/bge-small-zh-v1.5"
    vector_db_path = "./vector_db"
    chunk_size = 500
    chunk_overlap = 50

    # ========== 蒸馏工具 ==========
    distill_output_file = "distillation.txt"
    distill_deepseek_api_key = "your-deepseek-api-key"
    distill_qwen_api_key = "your-qwen-api-key"
    distill_default_topic = "共产主义"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    temperature = 0.8
    top_k = 50

    # ========== 2026 前沿新技术开关（全部默认关闭） ==========
    # MoE 相关
    use_latent_moe = False              # LatentMoE（低维潜空间路由，激活更多专家）
    use_intra_expert_sparsity = False   # 专家内稀疏（跳过80%静默神经元）
    intra_expert_sparsity_ratio = 0.2   # 保留比例 (0.2 即跳过 80%)
    use_budgeted_lora = False           # 预算化 LoRA（弹性低秩动态门控）
    budgeted_lora_rank = 4              # LoRA rank (4/8/16)
    
    # 推理优化
    use_tts = False                     # Test‑Time Scaling（多步自我反思）
    tts_steps = 3                       # 反思步数
    use_adaptive_compute = False        # 自适应推理（根据置信度决定是否继续反思）
    adaptive_compute_threshold = 0.8    # 置信度阈值
    use_early_exit = False              # 早退推理（中间层置信足够高即退出）
    early_exit_threshold = 0.9          # 早退置信度阈值
    
    # 长上下文优化
    use_kv_compress = False             # KV Cache 压缩（每隔 k 步采样）
    kv_compress_ratio = 0.25            # 压缩比例（保留 1/4）
    use_paged_attn = False              # PagedAttention 模拟（分页缓存，实验性）
    use_speed = False                   # SPEED: 层级非对称KV可见性（Prefill浅/Decode深）
    speed_prefill_layers = 0.75         # Prefill阶段保留的层比例
    
    # 注意力加速
    use_flash_attn3 = False             # 模拟 FlashAttention 3（FP8 + 分块）
    use_dga = False                     # 动态门控注意力 (Dynamic Gated Attention)
    use_uniprefill = False              # UniPrefill 预填充加速
    
    # 弹性推理
    use_star_elastic = False            # Star Elastic（一个模型内嵌多个子模型，动态选择）
    elastic_sizes = "4,6,8"              # 子模型的层数（字符串，如 "4,6,8"）
    
    # 隐式推理
    use_latent_reasoning = False        # 隐式推理模块（压缩序列为隐向量）
    latent_reasoning_dim = 128          # 隐向量维度
    
    # 其他增强
    use_adaptive_moe_load = False       # 动态调整专家内部稀疏比例（根据负载）
    use_mtp_distillation = False        # MTP 蒸馏（预训练阶段辅助）

    def validate(self):
        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.max_seq_len > 0
        if self.use_moe:
            assert self.num_experts >= self.top_k_experts
            if self.moe_n_groups > 1:
                assert self.num_experts % self.moe_n_groups == 0
        if self.attn_type in ("mla",):
            assert self.mla_q_lora_rank > 0 and self.mla_kv_lora_rank > 0
        if self.attn_type in ("swa", "hybrid", "hybrid_gdn"):
            assert self.swa_window_size > 0