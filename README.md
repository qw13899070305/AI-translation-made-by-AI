# 🤖 我的专属 AI 助手 · My Own AI Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <strong>从零手写的多模态对话模型 · 融合 DeepSeek-V3、MiMo-V2-Flash 与 Qwen3-Coder 前沿架构</strong>
  <br>
  <em>A multimodal conversational AI built from scratch, integrating cutting-edge architectures from DeepSeek-V3, MiMo-V2-Flash and Qwen3-Coder.</em>
</p>

<details>
<summary><strong>🇨🇳 中文介绍 · Click to expand</strong></summary>

## ✨ 核心特性

| 功能 | 说明 |
| :--- | :--- |
| 🧠 **混合注意力架构** | 支持 GQA / MLA / SWA / Hybrid 四种模式自由切换，参考 MiMo-V2-Flash 的 5:1 滑动窗口与全局注意力混合比 |
| ⚡ **增强 MoE 架构** | Sigmoid 门控 + 共享专家 + 分组限制 + 动态偏置负载均衡，参考 DeepSeek-V3 的 Auxiliary-Loss-Free 策略 |
| 🎤 **多模态输入** | 支持文本、图片、语音、PDF/TXT 文档上传 |
| 📚 **私有知识库 (RAG)** | 上传文档即可问答，AI 基于文档内容精准回答 |
| 🧬 **LoRA 高效微调** | 仅训练 0.5% 参数，支持权重保留的 LoRA 线性层替换 |
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
- 📦 **训练模型**
- 🔧 **训练分词器**
- 🧰 **独立工具**（联网搜索、导出历史、模型量化、数据预览、记忆测试、**数据扩充、多任务学习、持续学习**）

## 📁 项目结构

### 核心模块
| 文件 | 用途 |
| :--- | :--- |
| `config.py` | 全局配置（模型参数、数据集、训练超参） |
| `model.py` | 核心 Transformer（GQA / MLA / SWA / Hybrid + 增强 MoE + MTP） |
| `train.py` | 训练脚本（支持 LoRA、早停） |
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
| `config_manager.py` | **交互式配置管理中心**（性能优化、网络镜像、数据集管理、参数校验） |

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

## 🌟 架构亮点

- **混合注意力机制**：参考 MiMo-V2-Flash 的 SWA + 全局注意力 5:1 混合比例，支持 GQA / MLA / SWA / Hybrid 四种模式自由切换
- **增强 MoE 架构**：参考 DeepSeek-V3 的 Sigmoid 门控 + 分组限制 + 共享专家 + Auxiliary-Loss-Free 动态偏置负载均衡
- **多 Token 预测 (MTP)**：参考 MiMo-V2-Flash 的 3 层轻量级预测头，支持自投机解码加速
- **多教师在线蒸馏 (MOPD)**：参考 MiMo-V2-Flash 的多领域专家教师模型蒸馏策略
- **真正的端到端多模态设计**：图文语音一站式处理，自研 Q-Former + CLIP 视觉编码
- **开箱即用**：跨平台安装脚本，零门槛上手
- **丰富的扩展工具**：联网搜索、对话导出、模型量化、数据扩充、多任务学习、持续学习等即插即用
- **学习价值极高**：完整展示现代 LLM 核心技术（LoRA、RAG、多模态、持续学习、知识蒸馏）

## 🧠 技术参考

本项目架构设计参考了以下开源项目：
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)：MoE 门控与负载均衡策略
- [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)：滑动窗口注意力、MTP、MOPD 蒸馏
- [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder)：混合注意力与超稀疏 MoE

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源，欢迎自由使用、修改和分发。

</details>

<details>
<summary><strong>🇬🇧 English · Click to expand</strong></summary>

## ✨ Core Features

| Feature | Description |
| :--- | :--- |
| 🧠 **Hybrid Attention** | GQA / MLA / SWA / Hybrid modes, with MiMo-V2-Flash's 5:1 SWA-to-Global ratio |
| ⚡ **Enhanced MoE** | Sigmoid gating + shared experts + group constraints + dynamic bias load balancing (DeepSeek-V3 style) |
| 🎤 **Multimodal Input** | Text, image, voice, PDF/TXT document uploads |
| 📚 **Private Knowledge Base (RAG)** | Upload documents and ask questions based on their content |
| 🧬 **LoRA Fine‑tuning** | Train only 0.5% of parameters with weight-preserving LoRA layer replacement |
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
- 📦 **Train Model**
- 🔧 **Train Tokenizer**
- 🧰 **Standalone Tools** (web search, history export, quantization, data preview, memory test, **data expansion, multi‑task learning, continual learning**)

## 📁 Project Structure

### Core Modules
| File | Purpose |
| :--- | :--- |
| `config.py` | Global configuration |
| `model.py` | Core Transformer (GQA / MLA / SWA / Hybrid + Enhanced MoE + MTP) |
| `train.py` | Training script (LoRA support, early stopping) |
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
| `enhanced_data_loader.py` | **Data expansion (auto‑merge local corpora)** |
| `multitask_trainer.py` | **Multi‑task learning trainer** |
| `continual_trainer.py` | **Continual learning trainer** |
| `distill_mopd.py` | **MOPD multi‑teacher distillation trainer** |

### Configuration & Management
| File | Purpose |
| :--- | :--- |
| `config_manager.py` | **Interactive configuration center** (performance, mirror, datasets, validation) |

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

## 🌟 Architecture Highlights

- **Hybrid Attention Mechanisms**: MiMo-V2-Flash's SWA + Global attention with 5:1 ratio; supports GQA / MLA / SWA / Hybrid modes
- **Enhanced MoE Architecture**: DeepSeek-V3's Sigmoid gating + group constraints + shared experts + Auxiliary-Loss-Free dynamic bias load balancing
- **Multi‑Token Prediction (MTP)**: MiMo-V2-Flash's 3‑layer lightweight prediction heads for speculative decoding
- **Multi‑Teacher On‑Policy Distillation (MOPD)**: MiMo-V2-Flash's domain‑expert teacher distillation strategy
- **Truly end‑to‑end multimodal**: Seamless text, image, and voice processing with custom Q‑Former + CLIP vision encoder
- **Ready out‑of‑the‑box**: Cross‑platform install scripts for zero‑friction setup
- **Rich tooling**: Web search, history export, quantization, data expansion, multi‑task learning, continual learning
- **Great learning resource**: Demonstrates modern LLM techniques (LoRA, RAG, multimodal, continual learning, knowledge distillation)

## 🧠 Technical References

This project's architecture draws inspiration from:
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3): MoE gating & load balancing strategies
- [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash): Sliding window attention, MTP, MOPD distillation
- [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder): Hybrid attention & ultra‑sparse MoE

## 📄 License

This project is open‑sourced under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

</details>