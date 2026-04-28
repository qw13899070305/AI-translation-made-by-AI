# 🤖 我的专属 AI 助手 · My Own AI Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <strong>从零手写的多模态对话模型 · 融合 DeepSeek V4、Qwen3-Next 与 MiMo-V2-Flash 前沿架构</strong>
  <br>
  <em>A multimodal conversational AI built from scratch, integrating cutting-edge architectures from DeepSeek V4, Qwen3-Next and MiMo-V2-Flash.</em>
</p>

<details>
<summary><strong>🇨🇳 中文介绍 · Click to expand</strong></summary>

## ✨ 核心特性

| 功能 | 说明 |
| :--- | :--- |
| 🧠 **七大注意力模式** | GQA / MLA / SWA / CSA (压缩稀疏注意力) / Gated DeltaNet (线性注意力) / Hybrid / Hybrid-GDN，一键切换 |
| ⚡ **高稀疏度 MoE** | 32 专家 Top-2 路由 + 共享专家 + 动态偏置负载均衡，参考 DeepSeek V4 的 Auxiliary-Loss-Free 策略 |
| 🎤 **多模态输入** | 支持文本、图片、语音、PDF/TXT 文档上传 |
| 📚 **私有知识库 (RAG)** | 上传文档即可问答，AI 基于文档内容精准回答 |
| 🧬 **多 Token 预测 (MTP)** | 3 层预测头，支持自投机解码加速推理 |
| 🔬 **Muon 优化器** | 可选 Muon 优化器 (DeepSeek V4 / Kimi K2 标配) |
| 💾 **长期记忆** | 跨会话记住用户偏好和历史对话 |
| 🎭 **人设定制** | 通过 `persona.txt` 一键定义 AI 性格与语气 |
| ⚡ **模型量化** | 一键压缩为 INT8，推理速度提升 50% |
| 🌐 **联网搜索** | 独立工具，支持 DuckDuckGo 实时搜索 |
| 📜 **对话导出** | 导出为 Markdown / JSON / TXT，方便保存与分享 |
| 🖥️ **全平台支持** | Linux · macOS · Windows，提供一键安装与启动脚本 |

## 🆕 新增训练增强工具

| 工具 | 功能 | 使用方式 |
| :--- | :--- | :--- |
| 📦 **数据扩充** | 自动下载及合并百度百科、LCCC对话、豆瓣评论、微博语料等多源高质量数据，导出为训练文本 | 启动菜单 → 独立工具 → 6 |
| 🔄 **多任务学习** | 同时训练语言建模、问答、情感分析等多个任务，提升泛化能力 | 启动菜单 → 独立工具 → 7 |
| ⏳ **持续学习** | 带经验回放的增量学习，学习新任务时不会遗忘旧知识 | 启动菜单 → 独立工具 → 8 |
| 🎓 **MOPD 多教师蒸馏** | 支持多领域专家教师模型在线策略蒸馏，从大模型高效迁移知识 | 独立脚本 `distill_mopd.py` |

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
- 📚 **RAG 模块测试**
- 🎭 **人设对话模式**
- 📦 **训练模型**（支持 Muon 优化器 / AMP 混合精度）
- 🔧 **训练分词器**
- 🧰 **独立工具**（联网搜索、导出历史、模型量化、数据预览、记忆测试、**数据扩充、多任务学习、持续学习、MOPD 蒸馏**）

## 📁 项目结构

### 核心模块
| 文件 | 用途 |
| :--- | :--- |
| `config.py` | 全局配置（模型参数、数据集、训练超参） |
| `model.py` | 核心 Transformer（GQA / MLA / SWA / Hybrid + 增强 MoE + MTP） |
| `train.py` | 训练脚本（支持 LoRA、早停、Muon、AMP） |
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
| `rag_module.py` | RAG 私有知识库检索 |
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
- **QwenStyleMoE**：高稀疏度 + 共享专家 + 动态偏置，可配置 32‑256 个专家
- **多 Token 预测 (MTP)**：3 层轻量级预测头，支持自投机解码加速
- **多教师在线蒸馏 (MOPD)**：参考 MiMo 蒸馏策略，支持 DeepSeek / Qwen 等远端教师
- **Muon 优化器**：可选 Muon 优化器，训练收敛更快
- **混合精度训练 (AMP)**：自动启用，显存降低 40%
- **NTK‑aware RoPE**：零成本扩展上下文窗口（512 → 2048+）
- **全平台中英双语**：安装、启动、配置、蒸馏工具均支持中英双语

## 🧠 技术参考

本项目架构设计参考了以下开源项目：
- [DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V3)：混合稀疏注意力 (CSA+HCA)、MoE 门控与负载均衡策略
- [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)：滑动窗口注意力、MTP、MOPD 蒸馏
- [Qwen3-Next](https://github.com/QwenLM/Qwen3-Coder)：Gated DeltaNet 与超稀疏 MoE
- [Kimi K2.6](https://www.moonshot.cn/)：万亿 MoE、Muon 优化器、MLA 注意力

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源，欢迎自由使用、修改和分发。

</details>

<details>
<summary><strong>🇬🇧 English · Click to expand</strong></summary>

## ✨ Core Features

| Feature | Description |
| :--- | :--- |
| 🧠 **7 Attention Modes** | GQA / MLA / SWA / CSA / Gated DeltaNet / Hybrid / Hybrid-GDN |
| ⚡ **High-Sparsity MoE** | 32 experts Top-2 routing + shared expert + dynamic bias (DeepSeek V4 style) |
| 🎤 **Multimodal Input** | Text, image, voice, PDF/TXT document uploads |
| 📚 **Private Knowledge Base (RAG)** | Upload documents and ask questions based on their content |
| 🧬 **Multi-Token Prediction (MTP)** | 3‑layer prediction heads with speculative decoding (2‑3× speedup) |
| 🔬 **Muon Optimizer** | Optional Muon optimizer (adopted by DeepSeek V4 / Kimi K2) |
| 💾 **Long‑term Memory** | Remembers user preferences across sessions |
| 🎭 **Custom Persona** | Define AI personality via `persona.txt` |
| ⚡ **Model Quantization** | Compress to INT8 for 50% faster inference |
| 🌐 **Web Search** | Standalone DuckDuckGo search tool |
| 📜 **Conversation Export** | Export history as Markdown / JSON / TXT |
| 🖥️ **Cross‑platform** | Linux · macOS · Windows with one‑click install scripts |

## 🆕 New Training Enhancement Tools

| Tool | Function | Usage |
| :--- | :--- | :--- |
| 📦 **Data Expansion** | Auto‑download & merge Baidu Baike, LCCC dialogues, Douban reviews, Weibo corpus, etc. | Start menu → Tools → 6 |
| 🔄 **Multi‑Task Learning** | Train on language modeling, QA, and sentiment analysis simultaneously | Start menu → Tools → 7 |
| ⏳ **Continual Learning** | Incremental learning with experience replay | Start menu → Tools → 8 |
| 🎓 **MOPD Distillation** | Multi‑teacher on‑policy distillation from domain‑expert models | Standalone script `distill_mopd.py` |

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
- 🛠️ **Configuration Center** (performance tuning, network mirror, dataset management, environment check)
- 🌐 **Web Interface** (Gradio)
- 💬 **Command‑line Chat**
- 🔌 **API Backend**
- 📚 **RAG Module Test**
- 🎭 **Persona Chat Mode**
- 📦 **Train Model** (supports Muon optimizer / AMP)
- 🔧 **Train Tokenizer**
- 🧰 **Standalone Tools** (web search, history export, quantization, data preview, memory test, **data expansion, multi‑task learning, continual learning, MOPD distillation**)

## 📁 Project Structure

### Core Modules
| File | Purpose |
| :--- | :--- |
| `config.py` | Global configuration |
| `model.py` | Core Transformer (GQA / MLA / SWA / Hybrid + Enhanced MoE + MTP) |
| `train.py` | Training script (LoRA support, early stopping, Muon, AMP) |
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
| `rag_module.py` | RAG retrieval module |
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
- **QwenStyleMoE**: High sparsity + shared expert + dynamic bias, configurable 32‑256 experts
- **Multi‑Token Prediction (MTP)**: 3‑layer prediction heads with speculative decoding
- **Multi‑Teacher On‑Policy Distillation (MOPD)**: Distillation from DeepSeek / Qwen etc.
- **Muon Optimizer**: Optional optimizer, adopted by DeepSeek V4 / Kimi K2
- **Mixed Precision Training (AMP)**: 40% less GPU memory
- **NTK‑aware RoPE**: Zero‑cost context extension (512 → 2048+)
- **Cross‑platform Bilingual**: Install, launch, configure, distillation tools all support Chinese & English

## 🧠 Technical References

This project's architecture draws inspiration from:
- [DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V3): Hybrid sparse attention (CSA+HCA), MoE gating & load balancing
- [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash): Sliding window attention, MTP, MOPD distillation
- [Qwen3-Next](https://github.com/QwenLM/Qwen3-Coder): Gated DeltaNet & ultra‑sparse MoE
- [Kimi K2.6](https://www.moonshot.cn/): Trillion‑scale MoE, Muon optimizer, MLA attention

## 📄 License

This project is open‑sourced under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

</details>