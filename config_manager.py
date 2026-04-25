#!/usr/bin/env python3
# config_manager.py —— 项目配置管理中心（交互式菜单版）
"""
用法：
  python config_manager.py            # 进入交互式菜单
  python config_manager.py auto       # 自动优化性能配置
  python config_manager.py check      # 检查训练环境
  python config_manager.py clean      # 清理缓存文件
"""

import os
import sys
import shutil
import subprocess
from config import Config

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
BACKUP_FILE = CONFIG_FILE + ".bak"

# ==================== 所有可配置参数定义 ====================
# (参数名, 默认值, 说明) 用于专家模式展示和 set_any_param 时的类型转换

ALL_CONFIG_PARAMS = [
    # 数据集
    ("text_datasets", ["Open-Orca/OpenOrca", "my_local_data.txt", "distillation.txt"], "训练数据集列表"),
    ("max_samples_per_dataset", 3000, "每个数据集最大样本数"),
    ("max_seq_len", 512, "最大序列长度"),
    ("multimodal_dataset", "liuhaotian/LLaVA-Instruct-150K", "多模态数据集"),
    ("use_multimodal", True, "是否启用多模态"),
    ("max_multimodal_samples", 1000, "多模态最大样本数"),
    # 分词器
    ("vocab_size", 4096, "词表大小"),
    ("tokenizer_prefix", "our_bpe", "分词器前缀"),
    # 模型架构
    ("dim", 384, "隐藏层维度"),
    ("n_layers", 8, "Transformer 层数"),
    ("n_heads", 8, "注意力头数"),
    ("n_kv_heads", 4, "KV 头数 (GQA)"),
    ("use_moe", True, "是否启用 MoE"),
    ("num_experts", 8, "MoE 专家总数"),
    ("top_k_experts", 2, "每 Token 激活的专家数"),
    # MoE 增强参数
    ("moe_use_sigmoid_gate", True, "使用 Sigmoid 门控 (否则 Softmax)"),
    ("moe_num_shared_experts", 1, "共享专家数量"),
    ("moe_n_groups", 2, "专家分组数"),
    ("moe_topk_group", 2, "选中的组数"),
    ("moe_norm_topk_prob", True, "归一化 topk 权重"),
    ("moe_routed_scaling_factor", 1.0, "路由专家缩放因子"),
    ("moe_use_aux_loss_free", True, "使用无辅助损失负载均衡"),
    # 位置编码
    ("rope_theta", 10000.0, "RoPE 基础角度"),
    ("rope_scaling_factor", 1.0, "RoPE 缩放因子"),
    # 注意力类型
    ("attn_type", "hybrid", "注意力类型: gqa / mla / swa / hybrid"),
    ("swa_window_size", 128, "SWA 滑动窗口大小"),
    ("swa_hybrid_ratio", 5, "SWA:Global 混合比例"),
    ("mla_q_lora_rank", 192, "MLA Q 压缩维度"),
    ("mla_kv_lora_rank", 128, "MLA KV 压缩维度"),
    ("mla_qk_rope_head_dim", 32, "MLA RoPE 维度"),
    ("mla_v_head_dim", 64, "MLA V 维度"),
    ("use_partial_rope", True, "部分维度应用 RoPE"),
    ("partial_rope_dim", 64, "RoPE 应用维度"),
    # MTP 多 Token 预测
    ("use_mtp", False, "是否启用 MTP"),
    ("mtp_num_layers", 3, "MTP 层数"),
    ("mtp_hidden_dim", 512, "MTP 隐藏维度"),
    # 多模态
    ("vision_encoder_name", "openai/clip-vit-base-patch32", "视觉编码器"),
    ("vision_embed_dim", 768, "视觉嵌入维度"),
    ("proj_dim", 384, "投影维度"),
    ("num_queries", 32, "Q-Former 查询数"),
    ("qformer_layers", 2, "Q-Former 层数"),
    ("qformer_heads", 8, "Q-Former 头数"),
    # LoRA
    ("use_lora", True, "是否启用 LoRA"),
    ("lora_r", 8, "LoRA rank"),
    ("lora_alpha", 32, "LoRA alpha"),
    ("lora_dropout", 0.1, "LoRA dropout"),
    ("lora_target_modules", ["wq", "wv", "kv_compress", "k_proj", "v_proj"], "LoRA 目标模块"),
    # 训练超参
    ("batch_size", 8, "批次大小"),
    ("learning_rate", 3e-4, "学习率"),
    ("epochs", 5, "训练轮数"),
    ("grad_clip", 1.0, "梯度裁剪阈值"),
    ("save_every", 1, "保存频率 (epoch)"),
    ("num_workers", 16, "数据加载子进程数"),
    # RAG
    ("embedding_model", "BAAI/bge-small-zh-v1.5", "嵌入模型"),
    ("vector_db_path", "./vector_db", "向量数据库路径"),
    ("chunk_size", 500, "文本块大小"),
    ("chunk_overlap", 50, "文本块重叠大小"),
    # 生成参数
    ("temperature", 0.8, "生成温度"),
    ("top_k", 50, "Top-K 采样"),
    # 设备
    ("device", "cuda" if __import__("torch").cuda.is_available() else "cpu", "计算设备"),
]

# ==================== 工具函数 ====================

def read_config_lines():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return f.readlines()

def write_config_lines(lines):
    if os.path.exists(CONFIG_FILE):
        shutil.copy(CONFIG_FILE, BACKUP_FILE)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

def set_config_param(key, value):
    """修改 config.py 中指定的参数"""
    lines = read_config_lines()
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key} = ") or stripped.startswith(f"{key}="):
            if isinstance(value, str):
                lines[i] = f"    {key} = \"{value}\"\n"
            elif isinstance(value, bool):
                lines[i] = f"    {key} = {value}\n"
            elif isinstance(value, list):
                lines[i] = f"    {key} = {value}\n"
            else:
                lines[i] = f"    {key} = {value}\n"
            found = True
            break
    if not found:
        print(f"⚠️ 未找到参数 {key}")
        return False
    write_config_lines(lines)
    print(f"✅ {key} → {value}")
    return True

def enable_dataset(name):
    """启用某个数据集"""
    lines = read_config_lines()
    inside_list = False
    for i, line in enumerate(lines):
        if "text_datasets = [" in line:
            inside_list = True
            continue
        if inside_list and "]" in line:
            inside_list = False
            continue
        if inside_list and name in line and line.strip().startswith("#"):
            lines[i] = line.replace("#", "").replace("  ", " ")
            if not lines[i].strip().endswith(","):
                lines[i] = lines[i].rstrip() + ",\n"
            write_config_lines(lines)
            print(f"✅ 已启用数据集: {name}")
            return
    print(f"⚠️ 未找到数据集: {name}")

def disable_dataset(name):
    """禁用某个数据集"""
    lines = read_config_lines()
    inside_list = False
    for i, line in enumerate(lines):
        if "text_datasets = [" in line:
            inside_list = True
            continue
        if inside_list and "]" in line:
            inside_list = False
            continue
        if inside_list and name in line and not line.strip().startswith("#"):
            lines[i] = "        # " + line.strip() + "\n"
            write_config_lines(lines)
            print(f"⏸️  已禁用数据集: {name}")
            return
    print(f"⚠️ 未找到数据集: {name}")

# ==================== 菜单模块 ====================

def auto_optimize():
    import multiprocessing
    import torch
    cpu_count = multiprocessing.cpu_count()
    print(f"\n🖥️  检测到 CPU 核心数: {cpu_count}")
    recommended_workers = min(cpu_count // 2, 16)
    print(f"推荐 num_workers: {recommended_workers}")
    ans = input("是否应用？(y/n): ").strip().lower()
    if ans == "y":
        set_config_param("num_workers", recommended_workers)
    current_bs = getattr(Config(), "batch_size", 8)
    print(f"当前 batch_size: {current_bs}")
    print("建议：显存充足可设为 16 或 32，显存不足保持 8 或更小")
    ans = input("是否修改？(输入新值或 n): ").strip().lower()
    if ans and ans != "n":
        set_config_param("batch_size", int(ans))
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        if gpu_mem > 8:
            print("💡 建议开启混合精度训练 (AMP)，可在 train.py 中手动添加。")

def menu_performance():
    while True:
        print("\n⚡ 性能优化设置")
        print("  1) 自动检测硬件并推荐配置")
        print("  2) 手动调整 batch_size")
        print("  3) 手动调整 num_workers")
        print("  4) 设置训练轮数 (epochs)")
        print("  5) 查看当前性能参数")
        print("  0) 返回主菜单")
        choice = input("请选择: ").strip()
        if choice == "1": auto_optimize()
        elif choice == "2":
            val = input("输入 batch_size: ").strip()
            if val: set_config_param("batch_size", int(val))
        elif choice == "3":
            val = input("输入 num_workers: ").strip()
            if val: set_config_param("num_workers", int(val))
        elif choice == "4":
            val = input("输入 epochs: ").strip()
            if val: set_config_param("epochs", int(val))
        elif choice == "5":
            cfg = Config()
            print(f"batch_size={cfg.batch_size}, num_workers={cfg.num_workers}, epochs={cfg.epochs}, lr={cfg.learning_rate}")
        elif choice == "0": break

def menu_network():
    print("\n🌐 网络与镜像设置")
    print("  1) 设置 HuggingFace 镜像 (加速下载)")
    print("  2) 清除 HuggingFace 镜像")
    print("  3) 查看当前镜像设置")
    choice = input("请选择: ").strip()
    if choice == "1":
        bashrc = os.path.expanduser("~/.bashrc")
        with open(bashrc, "r") as f: content = f.read()
        if "HF_ENDPOINT" not in content:
            with open(bashrc, "a") as f:
                f.write('\nexport HF_ENDPOINT=https://hf-mirror.com\n')
        print("✅ 已添加镜像到 ~/.bashrc，重启终端生效")
    elif choice == "2":
        bashrc = os.path.expanduser("~/.bashrc")
        with open(bashrc, "r") as f: lines = f.readlines()
        with open(bashrc, "w") as f:
            for line in lines:
                if "HF_ENDPOINT" not in line: f.write(line)
        print("✅ 已清除镜像设置")
    elif choice == "3":
        val = os.environ.get("HF_ENDPOINT", "未设置")
        print(f"当前镜像: {val}")

def menu_datasets():
    while True:
        cfg = Config()
        print("\n📚 数据集管理")
        for i, ds in enumerate(cfg.text_datasets):
            print(f"  {i+1}. {ds}")
        print("\n  a) 启用数据集  d) 禁用数据集  0) 返回")
        c = input("请选择: ").strip()
        if c == "0": break
        elif c == "a":
            name = input("数据集名: ").strip()
            if name: enable_dataset(name)
        elif c == "d":
            name = input("数据集名: ").strip()
            if name: disable_dataset(name)

def menu_environment():
    import torch
    print("\n🔍 环境检查\n")
    print(f"Python: {sys.version}")
    try:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    except: print("PyTorch 未安装")
    print(f"虚拟环境: {'是' if sys.prefix != sys.base_prefix else '否'}")
    tokenizer_path = "tokenizer/our_bpe.model"
    print(f"分词器: {'存在' if os.path.exists(tokenizer_path) else '缺失 (运行 tokenizer_train.py)'}")
    ckpt = len([f for f in os.listdir("checkpoints") if f.endswith(".pt")]) if os.path.exists("checkpoints") else 0
    lora = len([f for f in os.listdir("lora_weights") if f.endswith(".pt")]) if os.path.exists("lora_weights") else 0
    print(f"检查点: {ckpt} (全量) + {lora} (LoRA)")
    total, used, free = shutil.disk_usage(".")
    print(f"磁盘可用: {free/1024**3:.1f} GB")

def menu_clean():
    print("\n🧹 清理缓存")
    print("  1) HuggingFace 缓存")
    print("  2) __pycache__ 目录")
    print("  3) 所有 .pt 检查点")
    c = input("请选择: ").strip()
    if c == "1":
        d = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(d):
            size = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(d) for f in fs)/1024**3
            if input(f"确认删除 {size:.1f}GB 缓存？(y/n): ") == "y":
                shutil.rmtree(d); print("✅ 已清理")
    elif c == "2":
        n = 0
        for r,ds,_ in os.walk("."):
            if "__pycache__" in ds:
                shutil.rmtree(os.path.join(r,"__pycache__")); n+=1
        print(f"✅ 清理 {n} 个")
    elif c == "3":
        if input("⚠️ 确认删除所有 .pt 文件？(y/n): ") == "y":
            for r,_,fs in os.walk("."):
                for f in fs:
                    if f.endswith(".pt"): os.remove(os.path.join(r,f))
            print("✅ 已清理")

def expert_menu():
    while True:
        print("\n🧠 专家模式")
        print("  1) 查看全部参数")
        print("  2) 直接修改参数")
        print("  0) 返回")
        c = input(": ").strip()
        if c == "0": break
        elif c == "1":
            cfg = Config()
            for key, default, desc in ALL_CONFIG_PARAMS:
                val = getattr(cfg, key, "N/A")
                print(f"  {key:<30} = {str(val):<20} # {desc}")
        elif c == "2":
            key = input("参数名: ").strip()
            if not key: continue
            try:
                cur = getattr(Config(), key)
            except:
                print(f"❌ 参数 {key} 不存在")
                continue
            print(f"当前值: {cur}")
            val = input("新值: ").strip()
            if not val: continue
            # 类型转换
            info = [x for x in ALL_CONFIG_PARAMS if x[0]==key]
            if info:
                t = type(info[0][1])
                if t == bool: val = val.lower() in ("true","1","yes","on")
                elif t == list:
                    val = [x.strip() for x in val.split(",")]
                else:
                    try: val = t(val)
                    except: pass
            else:
                try: val = int(val)
                except:
                    try: val = float(val)
                    except:
                        if val.lower() in ("true","false"): val = val.lower()=="true"
            set_config_param(key, val)

# ==================== 主菜单 ====================

def main_menu():
    while True:
        print("\n" + "="*50)
        print("   🛠️  项目配置管理中心")
        print("="*50)
        print("  1) ⚡ 性能优化")
        print("  2) 🌐 网络镜像")
        print("  3) 📚 数据集管理")
        print("  4) 🔍 环境检查")
        print("  5) 🧹 清理缓存")
        print("  6) 📋 查看全部参数 (专家模式)")
        print("  7) ✏️  直接修改参数 (专家模式)")
        print("  8) 🔄 配置校验")
        print("  0) 退出")
        c = input("\n请选择: ").strip()
        if c == "1": menu_performance()
        elif c == "2": menu_network()
        elif c == "3": menu_datasets()
        elif c == "4": menu_environment()
        elif c == "5": menu_clean()
        elif c == "6":
            cfg = Config()
            for key, default, desc in ALL_CONFIG_PARAMS:
                val = getattr(cfg, key, "N/A")
                print(f"  {key:<30} = {str(val):<20} # {desc}")
        elif c == "7":
            expert_menu()
        elif c == "8":
            try:
                Config().validate()
                print("✅ 配置合法")
            except AssertionError as e:
                print(f"❌ {e}")
        elif c == "0":
            print("👋")
            break
        else:
            print("无效选择")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "auto": auto_optimize()
        elif sys.argv[1] == "check": menu_environment()
        elif sys.argv[1] == "clean": menu_clean()
        else: print("用法: python config_manager.py [auto|check|clean]")
    else:
        main_menu()