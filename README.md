# 我的专属 AI 助手

从零手写的多模态对话模型，集成了 LoRA、RAG、GQA、MoE 等前沿技术。

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 训练分词器：`python tokenizer_train.py`
3. 训练模型：`python train.py`
4. 启动 Web 界面：`python chat_web.py`
5. （可选）启动手机端 API：`cd backend && python main.py`

## 项目结构

- `config.py`：全局配置
- `model.py`：核心 Transformer 模型
- `train.py`：训练脚本
- `chat_web.py`：Gradio 网页界面
- `vision_module.py`：多模态视觉模块
- `rag_module.py`：RAG 检索模块
- `lora.py`：LoRA 高效微调

## 部署

支持 HuggingFace Spaces 一键部署，或使用 FastAPI 后端配合 Flutter 手机端。