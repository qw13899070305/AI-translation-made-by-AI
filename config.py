import torch

class Config:
    # ========== 数据 ==========
    text_datasets = [
        "Open-Orca/OpenOrca",
        "GAIR/MegaScience",
        "my_local_data.txt",
        "distillation.txt",
        "enhanced_data.txt"
    ]
    max_samples_per_dataset = 3000
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
    num_experts = 8
    top_k_experts = 2
    max_seq_len = 512
    rope_theta = 10000.0
    rope_scaling_factor = 1.0

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
    lora_target_modules = ["wq", "wv"]

    # ========== 训练 ==========
    batch_size = 8
    learning_rate = 3e-4
    epochs = 5
    grad_clip = 1.0
    save_every = 1
    checkpoint_dir = "./checkpoints"
    lora_checkpoint_dir = "./lora_weights"

    # ========== RAG ==========
    embedding_model = "BAAI/bge-small-zh-v1.5"
    vector_db_path = "./vector_db"
    chunk_size = 500
    chunk_overlap = 50

    # ========== 设备 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 生成 ==========
    temperature = 0.8
    top_k = 50

    # ----- 新增：参数合法性校验 -----
    def validate(self):
        assert self.dim % self.n_heads == 0, \
            f"dim ({self.dim}) 必须能被 n_heads ({self.n_heads}) 整除"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) 必须能被 n_kv_heads ({self.n_kv_heads}) 整除"
        assert self.max_seq_len > 0, "max_seq_len 必须为正数"
        if self.use_moe:
            assert self.num_experts >= self.top_k_experts, \
                f"num_experts ({self.num_experts}) 必须 >= top_k_experts ({self.top_k_experts})"