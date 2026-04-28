# 🤖 MiniChat · 我的专属 AI 助手

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg">
</p>

<p align="center">
  <strong>从零手写的多模态对话模型 · 融合 DeepSeek V4 / Qwen3-Next / MiMo-V2-Flash 前沿架构</strong><br>
  <em>A multimodal conversational AI built from scratch, integrating cutting-edge architectures from DeepSeek V4, Qwen3-Next and MiMo-V2-Flash.</em>
</p>

---

## ✨ 核心特性

| 功能 | 说明 |
| :--- | :--- |
| 🧠 **七大注意力模式** | GQA / MLA / SWA / CSA (压缩稀疏注意力) / Gated DeltaNet (线性注意力) / Hybrid / Hybrid-GDN, 一键切换 |
| ⚡ **高稀疏度 MoE** | 32 专家 Top-2 路由 + 共享专家 + 动态偏置负载均衡 (参考 DeepSeek V4) |
| 🧬 **多 Token 预测 (MTP)** | 3 层预测头，支持自投机解码，推理速度提升 2‑3 倍 |
| 📈 **NTK‑Aware RoPE** | 零成本扩展上下文窗口 (512 → 2048+ Token) |
| 🎤 **多模态输入** | 图片、语音、PDF/TXT 文档上传 + RAG 知识库问答 |
| 🧰 **丰富工具链** | 联网搜索、对话导出、模型量化、数据扩充、多任务学习、持续学习、多教师蒸馏 |
| 🔬 **Muon 优化器** | 可选 Muon 优化器 (DeepSeek V4 / Kimi K2 标配) |
| 🖥️ **全平台中英双语** | Linux · macOS · Windows，支持中英双语安装/启动/配置 |

---

## 🚀 快速开始

### 一键安装
| 平台 | 命令/操作 |
| :--- | :--- |
| Linux | `bash install.sh` |
| macOS | 双击 `install.command` |
| Windows | 双击 `install.bat` |

### 启动菜单
| 平台 | 命令/操作 |
| :--- | :--- |
| Linux/macOS | `bash start.sh` 或双击 `start.command` |
| Windows | 双击 `start.bat` 或右键 `start.ps1` → “使用 PowerShell 运行” |

启动后提供：
- 🛠️ 配置管理中心
- 🚀 Web 界面（Gradio）
- 💬 命令行对话 / 🎭 人设对话
- 🌐 API 后端服务
- 📚 RAG 知识库测试 / 📦 训练模型 / 🔧 训练分词器
- 🧰 独立工具菜单（联网搜索、导出、量化、多任务/持续学习、蒸馏等）

---

## 🧪 架构亮点 (2026)

- **混合注意力**：默认使用 `hybrid_gdn` 模式 (3 层 Gated DeltaNet + 1 层 GQA)，推理速度翻倍，内存更低
- **Compressed Sparse Attention (CSA)**：借鉴 DeepSeek V4，KV 缓存压缩 4x 并通过 Lightning Indexer 选 Top‑k
- **Gated DeltaNet**：线性注意力，参考 Qwen3-Next，适合超长文档
- **QwenStyleMoE**：高稀疏度 + 共享专家 + 动态偏置，可配置 32‑256 个专家
- **自投机解码**：MTP 模块与生成器集成，一次前向可接受多个 token
- **Muon 优化器**：训练时可选，更快收敛
- **混合精度训练 (AMP)**：显存降低 40%
- **NTK‑RoPE**：零成本长上下文扩展

---

## 🧠 技术参考 (2026)

| 模型 | 核心技术 | MiniChat 吸收点 |
| :--- | :--- | :--- |
| DeepSeek V4 | 混合稀疏注意力 (CSA+HCA)、FP4 权重、Muon 优化器 | CSA 注意力、MoE 动态偏置、NTK‑RoPE |
| Qwen3-Next | Gated DeltaNet、512 专家超稀疏 MoE | GatedDeltaNet、高稀疏路由 |
| MiMo-V2-Flash | SWA‑Global 混合注意力、MTP、MOPD 蒸馏 | Hybrid 模式、MTP、蒸馏工具 |
| Kimi K2.6 | 万亿 MoE、Muon 优化器、MLA 注意力 | QwenStyleMoE、MLA 注意力 |
| Nemotron 3 Super | 混合 Mamba‑Transformer + LatentMoE | 混合架构设计思想 |

---

## 📁 项目结构 (核心文件)
