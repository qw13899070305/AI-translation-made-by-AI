import torch

class Config:
    # ========== 数据 ==========
    text_datasets = [
        "Open-Orca/OpenOrca",        # 分词器训练需要的第一数据集（在线）
        "my_local_data.txt",
        "distillation.txt",
        "enhanced_data.txt"          # 你的扩充数据，等生成后自动可用
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
    num_experts = 64                # 提高至 64，模仿 Kimi K2 的极高稀疏度
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
    use_muon = False                   # 使用 Muon 优化器
    use_muon_clip = False              # 使用 MuonClip (Kimi K2 风格)
    muon_clip_grad = 1.0               # MuonClip 梯度裁剪阈值
    muon_clip_update = 1.0             # MuonClip 更新量裁剪阈值

    # ========== 训练增强 ==========
    use_fp8 = False                    # 动态 FP8 训练（需 GPU 支持）
    use_hta = False                    # HTA 学习率调度
    hta_warmup_steps = 100             # HTA warmup 步数

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