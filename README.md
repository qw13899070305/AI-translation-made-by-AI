# 🤖 我的专属 AI 助手 · My Own AI Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <strong>从零手写的多模态对话模型 · 集成 LoRA、RAG、GQA/MoE、语音输入</strong>
  <br>
  <em>A multimodal conversational AI built from scratch, featuring LoRA, RAG, GQA/MoE, and voice input.</em>
</p>

<details>
<summary><strong>🇨🇳 中文介绍 · Click to expand</strong></summary>

## ✨ 核心功能

| 功能 | 说明 |
| :--- | :--- |
| 🧠 **自研 Transformer 架构** | GQA 分组注意力 + MoE 混合专家 + RoPE 旋转位置编码，高效推理 |
| 🎤 **多模态输入** | 支持文本、图片、语音、PDF/TXT 文档上传 |
| 📚 **私有知识库 (RAG)** | 上传文档即可问答，AI 基于文档内容精准回答 |
| 🧬 **LoRA 高效微调** | 仅训练 0.5% 参数，快速适配新任务 |
| 💾 **长期记忆** | 跨会话记住用户偏好和历史对话 |
| 🎭 **人设定制** | 通过 `persona.txt` 一键定义 AI 性格与语气 |
| ⚡ **模型量化** | 一键压缩为 INT8，推理速度提升 50% |
| 🌐 **联网搜索** | 独立工具，支持 DuckDuckGo 实时搜索 |
| 📜 **对话导出** | 导出为 Markdown / JSON / TXT，方便保存与分享 |
| 🖥️ **全平台支持** | Linux · macOS · Windows，提供一键安装与启动脚本 |

## 🆕 新增训练增强工具

| 工具 | 功能 | 使用方式 |
| :--- | :--- | :--- |
| 📦 **数据扩充** | 自动下载维基百科、代码、对话等多源高质量数据，合并导出为训练文本 | 启动菜单 → 独立工具 → 6 |
| 🔄 **多任务学习** | 同时训练语言建模、问答、情感分析等多个任务，提升泛化能力 | 启动菜单 → 独立工具 → 7 |
| ⏳ **持续学习** | 带经验回放的增量学习，学习新任务时不会遗忘旧知识 | 启动菜单 → 独立工具 → 8 |

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
- 🌐 Web 界面（Gradio）
- 💬 命令行对话
- 🔌 API 后端服务
- 📚 RAG 模块测试
- 🎭 人设对话模式
- 🧰 独立工具（联网搜索、导出历史、模型量化、数据预览、记忆测试、**数据扩充、多任务学习、持续学习**）

## 📁 项目结构

### 核心模块
| 文件 | 用途 |
| :--- | :--- |
| `config.py` | 全局配置（模型参数、数据集、训练超参） |
| `model.py` | 核心 Transformer（GQA / MoE / RoPE） |
| `train.py` | 训练脚本（支持 LoRA、早停） |
| `dataset.py` | 数据加载器（支持本地文件与 HuggingFace 数据集） |
| `tokenizer_train.py` | 分词器训练 |
| `lora.py` | LoRA 微调模块 |
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
| `memory.py` | 长期记忆模块 |
| `tools.py` | 工具调用框架 |

### 独立工具
| 文件 | 用途 |
| :--- | :--- |
| `web_search.py` | 联网搜索 |
| `export_history.py` | 对话历史导出 |
| `quantize.py` | 模型量化 |
| `preview_data.py` | 数据集预览 |
| `recall.py` | 长期记忆测试 |
| `enhanced_data_loader.py` | **数据扩充（多源数据自动下载）** |
| `multitask_trainer.py` | **多任务学习训练器** |
| `continual_trainer.py` | **持续学习训练器** |

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

## 🌟 项目亮点

- **真正的端到端多模态设计** —— 图文语音一站式处理，非简单拼接
- **高效的推理架构** —— GQA + MoE 组合，显存占用降低约 50%
- **开箱即用** —— 跨平台安装脚本，零门槛上手
- **丰富的扩展工具** —— 联网搜索、对话导出、模型量化、数据扩充、多任务学习、持续学习等即插即用
- **学习价值极高** —— 完整展示现代 LLM 核心技术（LoRA、RAG、多模态、持续学习）

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源，欢迎自由使用、修改和分发。

</details>

<details>
<summary><strong>🇬🇧 English · Click to expand</strong></summary>

## ✨ Core Features

| Feature | Description |
| :--- | :--- |
| 🧠 **Custom Transformer** | GQA attention + MoE feedforward + RoPE positional encoding |
| 🎤 **Multimodal Input** | Text, image, voice, PDF/TXT document uploads |
| 📚 **Private Knowledge Base (RAG)** | Upload documents and ask questions based on their content |
| 🧬 **LoRA Fine‑tuning** | Train only 0.5% of parameters for fast task adaptation |
| 💾 **Long‑term Memory** | Remembers user preferences across sessions |
| 🎭 **Custom Persona** | Define AI personality via `persona.txt` |
| ⚡ **Model Quantization** | Compress to INT8 for 50% faster inference |
| 🌐 **Web Search** | Standalone DuckDuckGo search tool |
| 📜 **Conversation Export** | Export history as Markdown / JSON / TXT |
| 🖥️ **Cross‑platform** | Linux · macOS · Windows with one‑click install scripts |

## 🆕 New Training Enhancement Tools

| Tool | Function | Usage |
| :--- | :--- | :--- |
| 📦 **Data Expansion** | Automatically download high‑quality multi‑source data (Wikipedia, code, dialogues) and export as training text | Start menu → Tools → 6 |
| 🔄 **Multi‑Task Learning** | Train on language modeling, QA, and sentiment analysis simultaneously to improve generalization | Start menu → Tools → 7 |
| ⏳ **Continual Learning** | Incremental learning with experience replay; learns new tasks without forgetting old knowledge | Start menu → Tools → 8 |

## 📦 One‑Click Install

| Platform | Action |
| :--- | :--- |
| **Linux** | `bash install.sh` |
| **macOS** | Double‑click `install.command` |
| **Windows** | Double‑click `install.bat` |

The script automatically sets up the environment, installs dependencies, trains the tokenizer, and launches the AI.

## 🚀 Quick Start

If the environment is already configured, use the start scripts:

| Platform | Action |
| :--- | :--- |
| **Linux** | `bash start.sh` |
| **macOS** | Double‑click `start.command` |
| **Windows** | Right‑click `start.ps1` → Run with PowerShell |

The menu offers:
- 🌐 Web Interface (Gradio)
- 💬 Command‑line Chat
- 🔌 API Backend
- 📚 RAG Module Test
- 🎭 Persona Chat Mode
- 🧰 Standalone Tools (web search, export history, quantization, data preview, memory test, **data expansion, multi‑task learning, continual learning**)

## 📁 Project Structure

### Core Modules
| File | Purpose |
| :--- | :--- |
| `config.py` | Global configuration |
| `model.py` | Core Transformer (GQA / MoE / RoPE) |
| `train.py` | Training script (LoRA support, early stopping) |
| `dataset.py` | Data loader (local files & HuggingFace datasets) |
| `tokenizer_train.py` | Tokenizer training |
| `lora.py` | LoRA fine‑tuning module |
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
| `memory.py` | Long‑term memory module |
| `tools.py` | Tool calling framework |

### Standalone Tools
| File | Purpose |
| :--- | :--- |
| `web_search.py` | Web search |
| `export_history.py` | Conversation export |
| `quantize.py` | Model quantization |
| `preview_data.py` | Dataset preview |
| `recall.py` | Memory test |
| `enhanced_data_loader.py` | **Data expansion (auto‑download multi‑source data)** |
| `multitask_trainer.py` | **Multi‑task learning trainer** |
| `continual_trainer.py` | **Continual learning trainer** |

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

## 🌟 Highlights

- **Truly end‑to‑end multimodal** — seamless text, image, and voice processing
- **Efficient inference** — GQA + MoE reduces memory footprint by ~50%
- **Ready out‑of‑the‑box** — cross‑platform install scripts for zero‑friction setup
- **Rich tooling** — web search, history export, quantization, data expansion, multi‑task learning, continual learning
- **Great learning resource** — demonstrates modern LLM techniques (LoRA, RAG, multimodal, continual learning)

## 📄 License

This project is open‑sourced under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

</details>