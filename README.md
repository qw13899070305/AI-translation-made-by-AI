# 🤖 我的专属 AI 助手

从零手写的多模态对话模型，集成 LoRA、RAG、GQA/MoE、长期记忆、语音输入。

## 一键安装
| 平台 | 命令 |
|:---|:---|
| Linux | `bash install.sh` |
| macOS | 双击 `install.command` |
| Windows | 双击 `install.bat` |

## 快速启动
| 平台 | 命令 |
|:---|:---|
| Linux | `bash start.sh` |
| macOS | 双击 `start.command` |
| Windows | 右键 `start.ps1` → PowerShell 运行 |

启动后可选：Web界面、命令行、API、RAG测试、人设对话、独立工具。

## 核心文件
- `config.py` 全局配置
- `model.py` Transformer 核心（GQA/MoE/RoPE）
- `train.py` 训练脚本（支持 LoRA）
- `chat_web.py` Gradio Web 界面
- `chat_cli.py` 命令行对话
- `persona_chat.py` 人设对话
- `rag_module.py` RAG 检索
- `vision_module.py` 多模态视觉
- `memory.py` 长期记忆

## 独立工具
- `web_search.py` 联网搜索
- `export_history.py` 对话导出
- `quantize.py` 模型量化
- `preview_data.py` 数据预览
- `recall.py` 记忆测试

## 许可证
MIT