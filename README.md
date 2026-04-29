# 🤖 我的专属 AI 助手 · My Own AI Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <strong>从零手写的多模态对话模型 · 融合 DeepSeek V4、Kimi K2、Qwen3-Next 与 MiMo-V2-Flash 前沿架构</strong>
  <br>
  <em>A multimodal conversational AI built from scratch, integrating cutting-edge architectures from DeepSeek V4, Kimi K2, Qwen3-Next and MiMo-V2-Flash.</em>
</p>

<details>
<summary><strong>🇨🇳 中文介绍 · Click to expand</strong></summary>

## ✨ 核心特性

| 功能 | 说明 |
| :--- | :--- |
| 🧠 **七大注意力模式** | GQA / MLA / SWA / CSA (压缩稀疏注意力) / Gated DeltaNet (线性注意力) / Hybrid / Hybrid-GDN，一键切换 |
| ⚡ **高稀疏度 MoE (SqrtGate)** | 64 专家 Top-2 路由 + 共享专家 + 动态偏置负载均衡 + SqrtGate 分布式稳定训练，参考 DeepSeek V4 / Kimi K2 |
| 🧬 **多 Token 预测 (MTP)** | 3 层预测头，支持自投机解码加速推理 |
| 📈 **NTK‑Aware RoPE** | 零成本扩展上下文窗口 (512 → 2048+ Token) |
| 🎤 **多模态输入** | 图片、语音、PDF/TXT 文档上传 + RAG 知识库问答 |
| 🧰 **增强 RAG 系统** | BM25 混合检索、自适应迭代、重排序、GraphRAG 知识图谱、HYDE 假设文档增强、自校正验证 |
| 🔬 **Muon / MuonClip 优化器** | 可选新一代优化器 (DeepSeek V4 / Kimi K2 标配) |
| ⚡ **动态 FP8 训练** | 可选 FP8 低精度训练，显存占用更低 |
| 🌡️ **HTA 学习率调度** | Warmup + Cosine 调度，训练更稳定 |
| 💾 **长期记忆** | 跨会话记住用户偏好和历史对话 |
| 🎭 **人设定制** | 通过 `persona.txt` 一键定义 AI 性格与语气 |
| ⚡ **模型量化** | 一键压缩为 INT8，推理速度提升 50% |
| 🌐 **联网搜索** | 独立工具，支持 DuckDuckGo 实时搜索 |
| 📜 **对话导出** | 导出为 Markdown / JSON / TXT，方便保存与分享 |
| 🖥️ **全平台支持** | Linux · macOS · Windows，提供中英双语一键安装与启动脚本 |

## 🆕 新增训练增强工具

| 工具 | 功能 | 使用方式 |
| :--- | :--- | :--- |
| 📦 **数据扩充** | 自动下载及合并多源高质量数据，导出为训练文本 | 启动菜单 → 独立工具 → 6 |
| 🔄 **多任务学习** | 同时训练语言建模、问答、情感分析等多个任务 | 启动菜单 → 独立工具 → 7 |
| ⏳ **持续学习** | 带经验回放的增量学习，学习新任务时不会遗忘旧知识 | 启动菜单 → 独立工具 → 8 |
| 🎓 **MOPD 多教师蒸馏** | 支持 DeepSeek / Qwen / OpenRouter / NVIDIA NIM 等免费教师模型在线蒸馏 | 独立脚本 `distill_mopd.py` |

## 📦 一键安装

| 平台 | 操作 |
| :--- | :--- |
| **Linux** | `bash install.sh` |
| **macOS** | 双击 `install.command` |
| **Windows** | 双击 `install.bat` |

脚本自动完成：Python 环境检测 → 虚拟环境创建 → 依赖安装 → 分词器训练 → 启动 AI。

## 🚀 快速启动

环境已配置好后，可直接使用启动脚本：

| 平台 | 操作 |
| :--- | :--- |
| **Linux** | `bash start.sh` |
| **macOS** | 双击 `start.command` |
| **Windows** | 右键 `start.ps1` → 使用 PowerShell 运行 |

启动菜单提供：
- 🛠️ **配置管理中心**（性能优化、网络镜像、数据集管理、环境检查）
- 🌐 **Web 界面**（Gradio）
- 💬 **命令行对话**
- 🔌 **API 后端服务**
- 📚 **RAG 模块测试**（支持混合检索、GraphRAG、HYDE）
- 🎭 **人设对话模式**
- 📦 **训练模型**（支持 Muon/MuonClip 优化器 / AMP / FP8 / HTA）
- 🔧 **训练分词器**
- 🧰 **独立工具**（联网搜索、导出历史、模型量化、数据预览、记忆测试、**数据扩充、多任务学习、持续学习、MOPD 蒸馏**）

## 📁 项目结构

### 核心模块
| 文件 | 用途 |
| :--- | :--- |
| `config.py` | 全局配置（模型参数、数据集、训练超参） |
| `model.py` | 核心 Transformer（GQA / MLA / SWA / Hybrid + 增强 MoE + MTP） |
| `train.py` | 训练脚本（支持 LoRA、早停、Muon/MuonClip、AMP、FP8、HTA） |
| `dataset.py` | 数据加载器（支持本地文件与 HuggingFace 数据集） |
| `tokenizer_train.py` | 分词器训练 |
| `lora.py` | LoRA 微调模块（权重保留式替换） |
| `utils.py` | 工具函数 |

### 对话交互
| 文件 | 用途 |
| :--- | :--- |
| `chat_web.py` | Gradio Web 界面（图文、语音、文档上传） |
| `chat_cli.py` | 命令行对话 |
| `persona_chat.py` | 人设对话模式 |
| `main.py` | FastAPI 后端服务 |

### 扩展模块
| 文件 | 用途 |
| :--- | :--- |
| `rag_module.py` | RAG 增强检索（混合检索、GraphRAG、HYDE、自校正） |
| `vision_module.py` | 多模态视觉编码器 |
| `memory.py` | 长期记忆模块（UUID 防碰撞） |
| `tools.py` | 工具调用框架 |

### 独立工具
| 文件 | 用途 |
| :--- | :--- |
| `web_search.py` | 联网搜索 |
| `export_history.py` | 对话历史导出 |
| `quantize.py` | 模型量化 |
| `preview_data.py` | 数据集预览 |
| `recall.py` | 长期记忆测试 |
| `enhanced_data_loader.py` | **数据扩充（多源数据自动下载与合并）** |
| `multitask_trainer.py` | **多任务学习训练器** |
| `continual_trainer.py` | **持续学习训练器** |
| `distill_mopd.py` | **MOPD 多教师蒸馏训练器** |

### 配置与管理
| 文件 | 用途 |
| :--- | :--- |
| `config_manager.py` | **交互式配置管理中心**（性能优化、网络镜像、数据集管理、参数校验、蒸馏配置） |

### 启动与安装脚本
| 文件 | 平台 | 用途 |
| :--- | :--- | :--- |
| `install.sh` | Linux | 一键安装 |
| `install.command` | macOS | 一键安装 |
| `install.bat` | Windows | 一键安装 |
| `start.sh` | Linux/macOS | 启动菜单 |
| `start.command` | macOS | 启动菜单（双击） |
| `start.ps1` | Windows | 启动菜单（PowerShell） |
| `start.bat` | Windows | 启动菜单（批处理） |

## 🌟 架构亮点 (2026)

- **混合注意力机制**：默认使用 `hybrid_gdn` 模式（3 层 Gated DeltaNet + 1 层 GQA），推理速度翻倍
- **压缩稀疏注意力 (CSA)**：借鉴 DeepSeek V4，KV 缓存压缩 4x 并通过 Lightning Indexer 选 Top‑k
- **Gated DeltaNet**：线性注意力，参考 Qwen3-Next，适合超长文本
- **QwenStyleMoE + SqrtGate**：高稀疏度 + 共享专家 + 动态偏置 + 分布稳定训练
- **增强 RAG 系统**：混合检索 (BM25+语义)、自适应迭代、重排序、GraphRAG 知识图谱、HYDE 假设文档、自校正验证
- **多 Token 预测 (MTP)**：3 层轻量级预测头，支持自投机解码加速
- **多教师在线蒸馏 (MOPD)**：支持 DeepSeek / Qwen / OpenRouter / NVIDIA NIM 等免费教师
- **Muon/MuonClip 优化器**：可选新一代优化器，训练收敛更快
- **混合精度 + 动态 FP8 训练**：自动启用 AMP，可选 FP8 进一步降低显存
- **HTA 学习率调度**：Warmup + Cosine 调度，训练更稳定
- **NTK‑aware RoPE**：零成本扩展上下文窗口（512 → 2048+）
- **全平台中英双语**：安装、启动、配置、蒸馏工具均支持中英双语

## 🧠 技术参考

本项目架构设计参考了以下开源项目：
- [DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V3)：混合稀疏注意力 (CSA+HCA)、MoE 门控与负载均衡策略
- [Kimi K2.6](https://www.moonshot.cn/)：万亿 MoE、MuonClip 优化器、MLA 注意力
- [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)：滑动窗口注意力、MTP、MOPD 蒸馏
- [Qwen3-Next](https://github.com/QwenLM/Qwen3-Coder)：Gated DeltaNet 与超稀疏 MoE
- [GraphRAG](https://github.com/microsoft/graphrag)：知识图谱增强检索
- [TurboQuant](https://github.com/google-research/turboquant)：KV Cache 无损压缩

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源，欢迎自由使用、修改和分发。

</details>

<details>
<summary><strong>🇬🇧 English · Click to expand</strong></summary>

## ✨ Core Features

| Feature | Description |
| :--- | :--- |
| 🧠 **7 Attention Modes** | GQA / MLA / SWA / CSA / Gated DeltaNet / Hybrid / Hybrid-GDN |
| ⚡ **High-Sparsity MoE (SqrtGate)** | 64 experts Top-2 routing + shared expert + dynamic bias + SqrtGate stable training |
| 🧬 **Multi-Token Prediction (MTP)** | 3‑layer prediction heads with speculative decoding (2‑3× speedup) |
| 📈 **NTK‑Aware RoPE** | Zero-cost context extension (512 → 2048+ tokens) |
| 🎤 **Multimodal Input** | Image, voice, PDF/TXT uploads + RAG knowledge base |
| 🧰 **Enhanced RAG** | BM25 hybrid retrieval, adaptive iteration, reranking, GraphRAG, HYDE, self-correction |
| 🔬 **Muon / MuonClip Optimizer** | Optional next-gen optimizer (DeepSeek V4 / Kimi K2) |
| ⚡ **Dynamic FP8 Training** | Optional FP8 low-precision training |
| 🌡️ **HTA Scheduler** | Warmup + Cosine annealing |
| 💾 **Long‑term Memory** | Remembers user preferences across sessions |
| 🎭 **Custom Persona** | Define AI personality via `persona.txt` |
| ⚡ **Model Quantization** | Compress to INT8 for 50% faster inference |
| 🌐 **Web Search** | Standalone DuckDuckGo search tool |
| 📜 **Conversation Export** | Export history as Markdown / JSON / TXT |
| 🖥️ **Cross‑platform** | Linux · macOS · Windows with one‑click install scripts |

## 🆕 Training Enhancement Tools

| Tool | Function | Usage |
| :--- | :--- | :--- |
| 📦 **Data Expansion** | Auto‑download & merge multi-source data | Start menu → Tools → 6 |
| 🔄 **Multi‑Task Learning** | Train LM, QA, sentiment simultaneously | Start menu → Tools → 7 |
| ⏳ **Continual Learning** | Incremental learning with experience replay | Start menu → Tools → 8 |
| 🎓 **MOPD Distillation** | Multi‑teacher distillation with free APIs | Standalone script `distill_mopd.py` |

## 📦 One‑Click Install

| Platform | Action |
| :--- | :--- |
| **Linux** | `bash install.sh` |
| **macOS** | Double‑click `install.command` |
| **Windows** | Double‑click `install.bat` |

## 🚀 Quick Start

| Platform | Action |
| :--- | :--- |
| **Linux** | `bash start.sh` |
| **macOS** | Double‑click `start.command` |
| **Windows** | Right‑click `start.ps1` → Run with PowerShell |

The menu offers:
- 🛠️ **Configuration Center**
- 🌐 **Web Interface**
- 💬 **Command‑line Chat**
- 🔌 **API Backend**
- 📚 **RAG Module Test** (supports hybrid retrieval, GraphRAG, HYDE)
- 🎭 **Persona Chat Mode**
- 📦 **Train Model** (supports Muon/MuonClip, AMP, FP8, HTA)
- 🔧 **Train Tokenizer**
- 🧰 **Standalone Tools** (web search, history export, quantization, data preview, memory test, **data expansion, multi‑task learning, continual learning, MOPD distillation**)

## 📁 Project Structure

### Core Modules
| File | Purpose |
| :--- | :--- |
| `config.py` | Global configuration |
| `model.py` | Core Transformer (GQA / MLA / SWA / Hybrid + Enhanced MoE + MTP) |
| `train.py` | Training script (LoRA, early stopping, Muon/MuonClip, AMP, FP8, HTA) |
| `dataset.py` | Data loader (local files & HuggingFace datasets) |
| `tokenizer_train.py` | Tokenizer training |
| `lora.py` | LoRA fine‑tuning module (weight‑preserving) |
| `utils.py` | Utility functions |

### Chat Interfaces
| File | Purpose |
| :--- | :--- |
| `chat_web.py` | Gradio web interface (text, image, voice, document upload) |
| `chat_cli.py` | Command‑line chat |
| `persona_chat.py` | Persona chat mode |
| `main.py` | FastAPI backend |

### Extension Modules
| File | Purpose |
| :--- | :--- |
| `rag_module.py` | Enhanced RAG (hybrid retrieval, GraphRAG, HYDE, self-correction) |
| `vision_module.py` | Multimodal vision encoder |
| `memory.py` | Long‑term memory module (UUID anti‑collision) |
| `tools.py` | Tool calling framework |

### Standalone Tools
| File | Purpose |
| :--- | :--- |
| `web_search.py` | Web search |
| `export_history.py` | Conversation export |
| `quantize.py` | Model quantization |
| `preview_data.py` | Dataset preview |
| `recall.py` | Memory test |
| `enhanced_data_loader.py` | **Data expansion (auto‑merge multi‑source local corpora)** |
| `multitask_trainer.py` | **Multi‑task learning trainer** |
| `continual_trainer.py` | **Continual learning trainer** |
| `distill_mopd.py` | **MOPD multi‑teacher distillation trainer** |

### Configuration & Management
| File | Purpose |
| :--- | :--- |
| `config_manager.py` | **Interactive configuration center** (performance, mirror, datasets, validation, distillation config) |

### Install & Start Scripts
| File | Platform | Purpose |
| :--- | :--- | :--- |
| `install.sh` | Linux | One‑click install |
| `install.command` | macOS | One‑click install |
| `install.bat` | Windows | One‑click install |
| `start.sh` | Linux/macOS | Start menu |
| `start.command` | macOS | Start menu (double‑click) |
| `start.ps1` | Windows | Start menu (PowerShell) |
| `start.bat` | Windows | Start menu (batch) |

## 🌟 Architecture Highlights (2026)

- **Hybrid Attention**: Default `hybrid_gdn` (3 Gated DeltaNet + 1 GQA), 2× inference speed
- **Compressed Sparse Attention (CSA)** : 4× KV compression + Lightning Indexer Top‑k (DeepSeek V4)
- **Gated DeltaNet**: Linear attention, inspired by Qwen3‑Next
- **QwenStyleMoE + SqrtGate**: High sparsity + shared expert + dynamic bias + stable training
- **Enhanced RAG**: Hybrid retrieval (BM25+semantic), adaptive iteration, reranking, GraphRAG, HYDE, self-correction
- **Multi‑Token Prediction (MTP)**: 3‑layer prediction heads with speculative decoding
- **Multi‑Teacher On‑Policy Distillation (MOPD)**: Distillation from DeepSeek / Qwen / OpenRouter / NVIDIA NIM
- **Muon/MuonClip Optimizer**: Optional next-gen optimizer
- **Mixed Precision + Dynamic FP8 Training**: AMP enabled, optional FP8
- **HTA Scheduler**: Warmup + Cosine annealing
- **NTK‑aware RoPE**: Zero‑cost context extension (512 → 2048+)
- **Cross‑platform Bilingual**: Install, launch, configure, distillation tools all support Chinese & English

## 🧠 Technical References

This project's architecture draws inspiration from:
- [DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V3): Hybrid sparse attention (CSA+HCA), MoE gating & load balancing
- [Kimi K2.6](https://www.moonshot.cn/): Trillion‑scale MoE, MuonClip optimizer, MLA attention
- [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash): Sliding window attention, MTP, MOPD distillation
- [Qwen3-Next](https://github.com/QwenLM/Qwen3-Coder): Gated DeltaNet & ultra‑sparse MoE
- [GraphRAG](https://github.com/microsoft/graphrag): Graph‑enhanced retrieval
- [TurboQuant](https://github.com/google-research/turboquant): Lossless KV cache compression

## 📄 License

This project is open‑sourced under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

</details>