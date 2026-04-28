#!/usr/bin/env python3
# config_manager.py —— 项目配置管理中心（中英双语版）
import os, sys, shutil, re, importlib
from config import Config

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
BACKUP_FILE = CONFIG_FILE + ".bak"

# 多语言文本
TEXTS = {
    "en": {
        "lang_select": "Please select language / 请选择语言:",
        "lang_prompt": "Enter choice (1 or 2): ",
        "config_center": "🛠️  Project Configuration Center",
        "menu_performance": "⚡ Performance Optimization",
        "menu_network": "🌐 Network Mirror",
        "menu_datasets": "📚 Dataset Management",
        "menu_environment": "🔍 Environment Check",
        "menu_clean": "🧹 Clean Cache",
        "menu_view_all": "📋 View All Parameters",
        "menu_edit": "✏️  Directly Modify Parameters",
        "menu_refresh": "🔄 Refresh Configuration",
        "menu_validate": "✅ Validate Configuration",
        "menu_distill": "🎓 Distillation Tool Config",
        "menu_exit": "Exit",
        "distill_title": "🎓 Distillation Tool Configuration",
        "distill_api": "Modify API Keys",
        "distill_output": "Modify Output File",
        "distill_topic": "Modify Default Topic",
        "distill_launch": "Launch Distillation Tool",
        "distill_return": "Return",
        "perf_title": "⚡ Performance Optimization Settings",
        "perf_auto": "Auto-detect hardware and recommend settings",
        "perf_batch": "Manually adjust batch_size",
        "perf_workers": "Manually adjust num_workers",
        "perf_epochs": "Set training epochs",
        "perf_view": "View current performance parameters",
        "perf_return": "Return to main menu",
        "network_title": "🌐 Network & Mirror Settings",
        "network_set": "Set HuggingFace mirror (accelerate downloads)",
        "network_clear": "Clear HuggingFace mirror",
        "network_view": "View current mirror setting",
        "datasets_title": "📚 Current Dataset List:",
        "datasets_hint": "To modify, use Expert Mode or directly edit config.py",
        "env_title": "🔍 Environment Check",
        "clean_title": "🧹 Clean Cache",
        "clean_hf": "HuggingFace Cache",
        "clean_pycache": "__pycache__ Directories",
        "clean_checkpoints": "All .pt Checkpoints",
        "param_header": "  {:<30} = {:<20} # {}",
        "edit_prompt_name": "Parameter name: ",
        "edit_prompt_value": "New value: ",
        "edit_not_found": "❌ Parameter {} does not exist",
        "edit_current": "Current value: {}",
        "edit_success": "✅ {} → {}",
        "refresh_done": "✅ Configuration refreshed.",
        "validate_pass": "✅ Configuration is valid.",
        "validate_fail": "❌ {}",
        "param_not_found": "⚠️  Parameter {} not found.",
        "invalid_choice": "Invalid choice.",
        "goodbye": "👋 Goodbye.",
        "auto_cpu": "🖥️  Detected CPU cores: {}",
        "auto_recommend": "Recommend num_workers: {}",
        "auto_apply": "Apply? (y/n): ",
        "auto_batch_current": "Current batch_size: {}",
        "auto_batch_hint": "Suggestion: If GPU memory is sufficient, set to 16 or 32; otherwise keep 8 or less.",
        "auto_batch_prompt": "Modify? (enter new value or n): ",
        "auto_gpu": "🎮 GPU: {} ({:.1f} GB)",
        "auto_gpu_hint": "💡 It is recommended to manually enable mixed precision training (AMP).",
    },
    "zh": {
        "lang_select": "请选择语言 / Please select language:",
        "lang_prompt": "请输入选择 (1 或 2): ",
        "config_center": "🛠️  项目配置管理中心",
        "menu_performance": "⚡ 性能优化",
        "menu_network": "🌐 网络镜像",
        "menu_datasets": "📚 数据集管理",
        "menu_environment": "🔍 环境检查",
        "menu_clean": "🧹 清理缓存",
        "menu_view_all": "📋 查看全部参数",
        "menu_edit": "✏️  直接修改参数",
        "menu_refresh": "🔄 刷新配置",
        "menu_validate": "✅ 配置校验",
        "menu_distill": "🎓 蒸馏工具配置",
        "menu_exit": "退出",
        "distill_title": "🎓 蒸馏工具配置",
        "distill_api": "修改 API 密钥",
        "distill_output": "修改输出文件",
        "distill_topic": "修改默认主题",
        "distill_launch": "启动蒸馏工具",
        "distill_return": "返回",
        "perf_title": "⚡ 性能优化设置",
        "perf_auto": "自动检测硬件并推荐配置",
        "perf_batch": "手动调整 batch_size",
        "perf_workers": "手动调整 num_workers",
        "perf_epochs": "设置训练轮数 (epochs)",
        "perf_view": "查看当前性能参数",
        "perf_return": "返回主菜单",
        "network_title": "🌐 网络与镜像设置",
        "network_set": "设置 HuggingFace 镜像 (加速下载)",
        "network_clear": "清除 HuggingFace 镜像",
        "network_view": "查看当前镜像设置",
        "datasets_title": "📚 当前数据集列表:",
        "datasets_hint": "如需修改，请使用专家模式或直接编辑 config.py",
        "env_title": "🔍 环境检查",
        "clean_title": "🧹 清理缓存",
        "clean_hf": "HuggingFace 缓存",
        "clean_pycache": "__pycache__ 目录",
        "clean_checkpoints": "所有 .pt 检查点",
        "param_header": "  {:<30} = {:<20} # {}",
        "edit_prompt_name": "参数名: ",
        "edit_prompt_value": "新值: ",
        "edit_not_found": "❌ 参数 {} 不存在",
        "edit_current": "当前值: {}",
        "edit_success": "✅ {} → {}",
        "refresh_done": "✅ 配置已刷新。",
        "validate_pass": "✅ 配置合法。",
        "validate_fail": "❌ {}",
        "param_not_found": "⚠️  参数 {} 未找到。",
        "invalid_choice": "无效选择。",
        "goodbye": "👋 再见。",
        "auto_cpu": "🖥️  检测到 CPU 核心数: {}",
        "auto_recommend": "推荐 num_workers: {}",
        "auto_apply": "是否应用？(y/n): ",
        "auto_batch_current": "当前 batch_size: {}",
        "auto_batch_hint": "建议：显存充足可设为 16 或 32，显存不足保持 8 或更小。",
        "auto_batch_prompt": "是否修改？(输入新值或 n): ",
        "auto_gpu": "🎮 GPU: {} ({:.1f} GB)",
        "auto_gpu_hint": "💡 建议手动开启混合精度训练 (AMP)。",
    },
}

def select_language():
    print("=" * 50)
    print("  🤖 My Own AI Assistant / 我的专属 AI 助手")
    print("=" * 50)
    print(f"  {TEXTS['en']['lang_select']}")
    print("  1) English")
    print("  2) 中文")
    choice = input(f"  {TEXTS['en']['lang_prompt']}").strip()
    return "zh" if choice == "2" else "en"

LANG = select_language()
T = TEXTS[LANG]

def refresh_config():
    import config
    importlib.reload(config)
    global Config
    from config import Config

# 参数定义（全部参数，包括最新架构参数）
ALL_CONFIG_PARAMS = [
    ("text_datasets", ["Open-Orca/OpenOrca", "my_local_data.txt", "distillation.txt"], "训练数据集列表" if LANG=="zh" else "Training dataset list"),
    ("max_samples_per_dataset", 50000, "每个数据集最大样本数" if LANG=="zh" else "Max samples per dataset"),
    ("max_seq_len", 512, "最大序列长度" if LANG=="zh" else "Max sequence length"),
    ("multimodal_dataset", "liuhaotian/LLaVA-Instruct-150K", "多模态数据集" if LANG=="zh" else "Multimodal dataset"),
    ("use_multimodal", True, "是否启用多模态" if LANG=="zh" else "Enable multimodal"),
    ("max_multimodal_samples", 1000, "多模态最大样本数" if LANG=="zh" else "Max multimodal samples"),
    ("vocab_size", 4096, "词表大小" if LANG=="zh" else "Vocabulary size"),
    ("tokenizer_prefix", "our_bpe", "分词器前缀" if LANG=="zh" else "Tokenizer prefix"),
    ("dim", 384, "隐藏层维度" if LANG=="zh" else "Hidden dimension"),
    ("n_layers", 8, "Transformer 层数" if LANG=="zh" else "Number of Transformer layers"),
    ("n_heads", 8, "注意力头数" if LANG=="zh" else "Number of attention heads"),
    ("n_kv_heads", 4, "KV 头数 (GQA)" if LANG=="zh" else "Number of KV heads (GQA)"),
    ("use_moe", True, "是否启用 MoE" if LANG=="zh" else "Enable MoE"),
    ("num_experts", 8, "MoE 专家总数" if LANG=="zh" else "Total MoE experts"),
    ("top_k_experts", 2, "每 Token 激活的专家数" if LANG=="zh" else "Experts activated per token"),
    ("moe_use_sigmoid_gate", True, "使用 Sigmoid 门控" if LANG=="zh" else "Use Sigmoid gate"),
    ("moe_num_shared_experts", 1, "共享专家数量" if LANG=="zh" else "Number of shared experts"),
    ("moe_n_groups", 2, "专家分组数" if LANG=="zh" else "Number of expert groups"),
    ("moe_topk_group", 2, "选中的组数" if LANG=="zh" else "Top-k groups selected"),
    ("moe_norm_topk_prob", True, "归一化 topk 权重" if LANG=="zh" else "Normalize topk probabilities"),
    ("moe_routed_scaling_factor", 1.0, "路由专家缩放因子" if LANG=="zh" else "Routed expert scaling factor"),
    ("moe_use_aux_loss_free", True, "使用无辅助损失负载均衡" if LANG=="zh" else "Use auxiliary-loss-free load balancing"),
    ("rope_theta", 10000.0, "RoPE 基础角度" if LANG=="zh" else "RoPE base theta"),
    ("rope_scaling_factor", 1.0, "RoPE 缩放因子" if LANG=="zh" else "RoPE scaling factor"),
    ("use_ntk_rope", True, "启用 NTK-RoPE" if LANG=="zh" else "Use NTK-RoPE"),
    ("target_context_len", 2048, "NTK-RoPE 目标上下文长度" if LANG=="zh" else "NTK-RoPE target context length"),
    ("attn_type", "hybrid", "注意力类型" if LANG=="zh" else "Attention type"),
    ("swa_window_size", 128, "SWA 滑动窗口大小" if LANG=="zh" else "SWA window size"),
    ("swa_hybrid_ratio", 5, "SWA:Global 混合比例" if LANG=="zh" else "SWA:Global hybrid ratio"),
    ("gdn_hybrid_ratio", 4, "Gated DeltaNet:GQA 混合比" if LANG=="zh" else "Gated DeltaNet:GQA ratio"),
    ("mla_q_lora_rank", 192, "MLA Q 压缩维度" if LANG=="zh" else "MLA Q LoRA rank"),
    ("mla_kv_lora_rank", 128, "MLA KV 压缩维度" if LANG=="zh" else "MLA KV LoRA rank"),
    ("mla_qk_rope_head_dim", 32, "MLA RoPE 维度" if LANG=="zh" else "MLA QK RoPE head dim"),
    ("mla_v_head_dim", 64, "MLA V 维度" if LANG=="zh" else "MLA V head dim"),
    ("use_partial_rope", True, "部分维度应用 RoPE" if LANG=="zh" else "Use partial RoPE"),
    ("partial_rope_dim", 64, "RoPE 应用维度" if LANG=="zh" else "Partial RoPE dim"),
    ("csa_compress_ratio", 4, "CSA 压缩比" if LANG=="zh" else "CSA compress ratio"),
    ("csa_top_k", 512, "CSA 稀疏 Top-k" if LANG=="zh" else "CSA sparse Top-k"),
    ("use_mtp", False, "是否启用 MTP" if LANG=="zh" else "Enable MTP"),
    ("mtp_num_layers", 3, "MTP 层数" if LANG=="zh" else "MTP number of layers"),
    ("mtp_hidden_dim", 512, "MTP 隐藏维度" if LANG=="zh" else "MTP hidden dimension"),
    ("use_mtp_speculative_decode", True, "推理时使用 MTP 自投机解码" if LANG=="zh" else "Use MTP speculative decoding in generation"),
    ("vision_encoder_name", "openai/clip-vit-base-patch32", "视觉编码器" if LANG=="zh" else "Vision encoder name"),
    ("vision_embed_dim", 768, "视觉嵌入维度" if LANG=="zh" else "Vision embed dimension"),
    ("proj_dim", 384, "投影维度" if LANG=="zh" else "Projection dimension"),
    ("num_queries", 32, "Q-Former 查询数" if LANG=="zh" else "Number of Q-Former queries"),
    ("qformer_layers", 2, "Q-Former 层数" if LANG=="zh" else "Q-Former layers"),
    ("qformer_heads", 8, "Q-Former 头数" if LANG=="zh" else "Q-Former heads"),
    ("use_lora", True, "是否启用 LoRA" if LANG=="zh" else "Enable LoRA"),
    ("lora_r", 8, "LoRA rank" if LANG=="zh" else "LoRA rank"),
    ("lora_alpha", 32, "LoRA alpha" if LANG=="zh" else "LoRA alpha"),
    ("lora_dropout", 0.1, "LoRA dropout" if LANG=="zh" else "LoRA dropout"),
    ("lora_target_modules", ["wq", "wv", "kv_compress", "k_proj", "v_proj"], "LoRA 目标模块" if LANG=="zh" else "LoRA target modules"),
    ("batch_size", 16, "批次大小" if LANG=="zh" else "Batch size"),
    ("learning_rate", 1e-4, "学习率" if LANG=="zh" else "Learning rate"),
    ("epochs", 20, "训练轮数" if LANG=="zh" else "Training epochs"),
    ("grad_clip", 1.0, "梯度裁剪阈值" if LANG=="zh" else "Gradient clip threshold"),
    ("save_every", 1, "保存频率 (epoch)" if LANG=="zh" else "Save frequency (epoch)"),
    ("num_workers", 16, "数据加载子进程数" if LANG=="zh" else "Number of data loading workers"),
    ("use_amp", True, "启用混合精度训练" if LANG=="zh" else "Use AMP"),
    ("use_muon", False, "使用 Muon 优化器" if LANG=="zh" else "Use Muon optimizer"),
    ("embedding_model", "BAAI/bge-small-zh-v1.5", "嵌入模型" if LANG=="zh" else "Embedding model"),
    ("vector_db_path", "./vector_db", "向量数据库路径" if LANG=="zh" else "Vector DB path"),
    ("chunk_size", 500, "文本块大小" if LANG=="zh" else "Chunk size"),
    ("chunk_overlap", 50, "文本块重叠大小" if LANG=="zh" else "Chunk overlap"),
    ("temperature", 0.8, "生成温度" if LANG=="zh" else "Generation temperature"),
    ("top_k", 50, "Top-K 采样" if LANG=="zh" else "Top-K sampling"),
    ("device", "cuda" if __import__("torch").cuda.is_available() else "cpu", "计算设备" if LANG=="zh" else "Device"),
    # 蒸馏工具配置
    ("distill_output_file", "distillation.txt", "蒸馏输出文件" if LANG=="zh" else "Distillation output file"),
    ("distill_deepseek_api_key", "your-deepseek-api-key", "DeepSeek API 密钥" if LANG=="zh" else "DeepSeek API key"),
    ("distill_qwen_api_key", "your-qwen-api-key", "Qwen API 密钥" if LANG=="zh" else "Qwen API key"),
    ("distill_default_topic", "共产主义", "默认蒸馏主题" if LANG=="zh" else "Default distillation topic"),
]

def read_config_lines():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f: return f.readlines()

def write_config_lines(lines):
    if os.path.exists(CONFIG_FILE): shutil.copy(CONFIG_FILE, BACKUP_FILE)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f: f.writelines(lines)

def set_config_param(key, value):
    lines = read_config_lines()
    found = False
    if isinstance(value, str): new_val_str = f'"{value}"'
    elif isinstance(value, bool): new_val_str = str(value)
    elif isinstance(value, list): new_val_str = str(value)
    else: new_val_str = str(value)
    pattern = re.compile(rf'^(\s*{re.escape(key)}\s*=\s*)(.+)', re.MULTILINE)
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            comment = ''
            if '#' in line: comment = '  # ' + line.split('#', 1)[1].strip()
            lines[i] = f'{m.group(1)}{new_val_str}{comment}\n'
            found = True
            break
    if not found: print(T["param_not_found"].format(key)); return False
    write_config_lines(lines); refresh_config()
    print(T["edit_success"].format(key, value)); return True

def auto_optimize():
    import multiprocessing, torch
    cpu_count = multiprocessing.cpu_count()
    print(T["auto_cpu"].format(cpu_count))
    recommended_workers = min(cpu_count // 2, 16)
    print(T["auto_recommend"].format(recommended_workers))
    ans = input(T["auto_apply"]).strip().lower()
    if ans == "y": set_config_param("num_workers", recommended_workers)
    current_bs = getattr(Config(), "batch_size", 8)
    print(T["auto_batch_current"].format(current_bs))
    print(T["auto_batch_hint"])
    ans = input(T["auto_batch_prompt"]).strip().lower()
    if ans and ans != "n":
        try: set_config_param("batch_size", int(ans))
        except: print(T["invalid_choice"])
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(T["auto_gpu"].format(gpu_name, gpu_mem))
        if gpu_mem > 8: print(T["auto_gpu_hint"])

def menu_performance():
    while True:
        print(f"\n{T['perf_title']}")
        print(f"  1) {T['perf_auto']}"); print(f"  2) {T['perf_batch']}"); print(f"  3) {T['perf_workers']}")
        print(f"  4) {T['perf_epochs']}"); print(f"  5) {T['perf_view']}"); print(f"  0) {T['perf_return']}")
        choice = input("  > ").strip()
        if choice == "1": auto_optimize()
        elif choice == "2":
            val = input("  batch_size: ").strip()
            if val:
                try: set_config_param("batch_size", int(val))
                except: print(T["invalid_choice"])
        elif choice == "3":
            val = input("  num_workers: ").strip()
            if val:
                try: set_config_param("num_workers", int(val))
                except: print(T["invalid_choice"])
        elif choice == "4":
            val = input("  epochs: ").strip()
            if val:
                try: set_config_param("epochs", int(val))
                except: print(T["invalid_choice"])
        elif choice == "5":
            refresh_config(); cfg = Config()
            print(f"  batch_size={cfg.batch_size}, num_workers={cfg.num_workers}, epochs={cfg.epochs}, lr={cfg.learning_rate}")
        elif choice == "0": break
        else: print(T["invalid_choice"])

def menu_network():
    print(f"\n{T['network_title']}")
    print(f"  1) {T['network_set']}"); print(f"  2) {T['network_clear']}"); print(f"  3) {T['network_view']}")
    choice = input("  > ").strip()
    if choice == "1":
        bashrc = os.path.expanduser("~/.bashrc")
        with open(bashrc, "r") as f: content = f.read()
        if "HF_ENDPOINT" not in content:
            with open(bashrc, "a") as f: f.write('\nexport HF_ENDPOINT=https://hf-mirror.com\n')
        print("✅ 镜像已添加")
    elif choice == "2":
        bashrc = os.path.expanduser("~/.bashrc")
        with open(bashrc, "r") as f: lines = f.readlines()
        with open(bashrc, "w") as f:
            for line in lines:
                if "HF_ENDPOINT" not in line: f.write(line)
        print("✅ 镜像已清除")
    elif choice == "3": print(f"当前镜像: {os.environ.get('HF_ENDPOINT', '未设置')}")

def menu_datasets():
    refresh_config(); cfg = Config()
    print(f"\n{T['datasets_title']}")
    for i, ds in enumerate(cfg.text_datasets): print(f"  {i+1}. {ds}")
    print(f"\n{T['datasets_hint']}")

def menu_environment():
    import torch
    print(f"\n{T['env_title']}\n")
    print(f"Python: {sys.version}")
    try:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("PyTorch 未安装")
    print(f"虚拟环境: {'是' if sys.prefix != sys.base_prefix else '否'}")
    tokenizer_path = "tokenizer/our_bpe.model"
    print(f"分词器: {'存在' if os.path.exists(tokenizer_path) else '缺失'}")
    ckpt = len([f for f in os.listdir("checkpoints") if f.endswith(".pt")]) if os.path.exists("checkpoints") else 0
    lora = len([f for f in os.listdir("lora_weights") if f.endswith(".pt")]) if os.path.exists("lora_weights") else 0
    print(f"检查点: {ckpt} (全量) + {lora} (LoRA)")
    total, used, free = shutil.disk_usage(".")
    print(f"磁盘可用: {free/1024**3:.1f} GB")

def menu_clean():
    print(f"\n{T['clean_title']}")
    print(f"  1) {T['clean_hf']}"); print(f"  2) {T['clean_pycache']}"); print(f"  3) {T['clean_checkpoints']}")
    c = input("  > ").strip()
    if c == "1":
        d = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(d) and input("  确认删除? (y/n): ") == "y": shutil.rmtree(d); print("✅ 已清理")
    elif c == "2":
        n = 0
        for r, ds, _ in os.walk("."):
            if "__pycache__" in ds: shutil.rmtree(os.path.join(r, "__pycache__")); n += 1
        print(f"✅ 清理 {n} 个")
    elif c == "3":
        if input("⚠️  确认删除所有 .pt 文件? (y/n): ") == "y":
            for r, _, fs in os.walk("."):
                for f in fs:
                    if f.endswith(".pt"): os.remove(os.path.join(r, f))
            print("✅ 已清理")

def menu_distill():
    print(f"\n{T['distill_title']}")
    print(f"  1) {T['distill_api']}"); print(f"  2) {T['distill_output']}"); print(f"  3) {T['distill_topic']}")
    print(f"  4) {T['distill_launch']}"); print(f"  0) {T['distill_return']}")
    sc = input("  > ").strip()
    if sc == "1":
        key = input("  输入密钥名称 (deepseek/qwen): ").strip()
        if key == "deepseek":
            val = input("  输入 DeepSeek API 密钥: ").strip()
            set_config_param("distill_deepseek_api_key", val)
        elif key == "qwen":
            val = input("  输入 Qwen API 密钥: ").strip()
            set_config_param("distill_qwen_api_key", val)
        else: print("  无效密钥名称")
    elif sc == "2":
        val = input("  输入输出文件名: ").strip()
        if val: set_config_param("distill_output_file", val)
    elif sc == "3":
        val = input("  输入默认主题: ").strip()
        if val: set_config_param("distill_default_topic", val)
    elif sc == "4":
        import subprocess
        refresh_config(); cfg = Config()
        topic = getattr(cfg, "distill_default_topic", "共产主义")
        output = getattr(cfg, "distill_output_file", "distillation.txt")
        subprocess.run([sys.executable, "distill_mopd.py", "--topic", topic, "--output", output, "--lang", LANG])
    elif sc == "0": pass
    else: print(T["invalid_choice"])

def manual_edit():
    key = input(T["edit_prompt_name"]).strip()
    if not key: return
    try:
        refresh_config(); cur = getattr(Config(), key)
    except: print(T["edit_not_found"].format(key)); return
    print(T["edit_current"].format(cur))
    val = input(T["edit_prompt_value"]).strip()
    if not val: return
    info = [x for x in ALL_CONFIG_PARAMS if x[0] == key]
    if info:
        t = type(info[0][1])
        if t == bool: val = val.lower() in ("true", "1", "yes", "on")
        elif t == list: val = [x.strip() for x in val.split(",")]
        else:
            try: val = t(val)
            except: pass
    else:
        try: val = int(val)
        except:
            try: val = float(val)
            except:
                if val.lower() in ("true", "false"): val = val.lower() == "true"
    set_config_param(key, val)

def main_menu():
    while True:
        print(f"\n{'='*50}")
        print(f"  {T['config_center']}")
        print(f"{'='*50}")
        print(f"  1) {T['menu_performance']}")
        print(f"  2) {T['menu_network']}")
        print(f"  3) {T['menu_datasets']}")
        print(f"  4) {T['menu_environment']}")
        print(f"  5) {T['menu_clean']}")
        print(f"  6) {T['menu_view_all']}")
        print(f"  7) {T['menu_edit']}")
        print(f"  8) {T['menu_refresh']}")
        print(f"  9) {T['menu_validate']}")
        print(f"  10) {T['menu_distill']}")
        print(f"  0) {T['menu_exit']}")
        c = input("\n  > ").strip()
        if c == "1": menu_performance()
        elif c == "2": menu_network()
        elif c == "3": menu_datasets()
        elif c == "4": menu_environment()
        elif c == "5": menu_clean()
        elif c == "6":
            refresh_config(); cfg = Config()
            for key, default, desc in ALL_CONFIG_PARAMS:
                val = getattr(cfg, key, "N/A")
                print(T["param_header"].format(key, str(val)[:20], desc))
        elif c == "7": manual_edit()
        elif c == "8": refresh_config(); print(T["refresh_done"])
        elif c == "9":
            refresh_config()
            try: Config().validate(); print(T["validate_pass"])
            except AssertionError as e: print(T["validate_fail"].format(e))
        elif c == "10": menu_distill()
        elif c == "0": print(T["goodbye"]); break
        else: print(T["invalid_choice"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "auto": auto_optimize()
        elif sys.argv[1] == "check": menu_environment()
        elif sys.argv[1] == "clean": menu_clean()
        else: print("用法: python config_manager.py [auto|check|clean]")
    else:
        main_menu()