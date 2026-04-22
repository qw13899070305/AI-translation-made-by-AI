🤖 我的专属 AI 助手 (MiniChat)

从零手写的多模态对话模型，集成了 LoRA 微调、RAG 检索、GQA/MoE 架构、长期记忆、语音输入、人设定制、模型量化等前沿技术。

✨ 核心特性

· 🧠 自研 Transformer 架构：GQA（分组查询注意力）+ MoE（混合专家）+ RoPE（旋转位置编码）
· 🎤 多模态输入：支持文本、图片、语音、文档上传
· 📚 RAG 私有知识库：上传 PDF/TXT，AI 基于文档内容回答
· 🧬 LoRA 高效微调：仅训练 0.5% 参数，快速适配新任务
· 💾 长期记忆：跨会话记住用户偏好和历史对话
· 🎭 人设定制：通过 persona.txt 自由定义 AI 性格
· ⚡ 模型量化：一键压缩为 INT8，推理速度提升 50%
· 🌐 跨平台启动：支持 Linux、macOS、Windows（批处理/PowerShell）
· 📱 移动端部署：兼容 Termux + proot-distro Debian 环境

---

📁 项目文件结构（完整版）

🎯 核心模块

文件 用途
config.py 全局配置中心：模型参数、数据集、训练超参数、路径等
model.py 核心 Transformer 模型：GQA 注意力、MoE 前馈、RMSNorm、RoPE
dataset.py 数据加载器：支持 HuggingFace 数据集和本地文本文件混合
train.py 训练脚本：支持全量训练和 LoRA 微调，自动保存检查点
tokenizer_train.py 分词器训练：基于 SentencePiece BPE 算法
lora.py LoRA 微调模块：低秩适配，高效微调大模型
utils.py 工具函数：格式化提示词、文件操作、下载等

💬 对话与交互

文件 用途
chat_web.py Gradio Web 界面：支持图文上传、语音输入、RAG 文档上传
chat_cli.py 命令行对话：纯终端交互，适合轻量使用
persona_chat.py 人设对话模式：独立脚本，从 persona.txt 读取人设并对话
main.py FastAPI 后端服务：提供 RESTful API，供手机端或第三方调用
app.py HuggingFace Spaces 部署入口：一键部署到公网

🧩 扩展模块

文件 用途
vision_module.py 多模态视觉编码器：CLIP + Q-Former，将图像转为模型可理解的向量
rag_module.py RAG 检索增强：基于 ChromaDB + LangChain，实现私有文档问答
memory.py 长期记忆模块：跨会话存储和检索历史对话
tools.py 工具调用框架：预留的天气、邮件等 API 调用接口

🛠️ 独立工具脚本

文件 用途
quantize.py 模型量化工具：将 FP32 模型压缩为 INT8，减小体积、提速
preview_data.py 数据集预览器：查看配置的数据集前几条样本
recall.py 长期记忆测试：独立测试记忆存储和检索功能
persona.txt 人设配置文件：定义 AI 的性格、语气、口头禅

🚀 启动脚本

文件 平台 用途
start.sh Linux Bash 启动脚本，彩色菜单，支持 6 种模式
start.command macOS 双击启动的终端脚本
start.bat Windows 批处理启动脚本
start.ps1 Windows PowerShell 启动脚本（功能完整版）

📋 配置文件

文件 用途
requirements.txt Python 依赖清单
my_local_data.txt 本地训练数据（可自定义）
distillation.txt 知识蒸馏数据集（可选）

---

🚀 快速开始

1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

2️⃣ 训练分词器

```bash
python tokenizer_train.py
```

3️⃣ 训练模型（可选，也可使用预训练权重）

```bash
python train.py
```

4️⃣ 启动服务

· Web 界面（推荐）：
  ```bash
  python chat_web.py
  ```
  浏览器访问 http://127.0.0.1:7860
· 命令行对话：
  ```bash
  python chat_cli.py
  ```
· 人设对话：
  ```bash
  python persona_chat.py
  ```
· API 后端：
  ```bash
  cd backend && python main.py
  ```

或直接运行启动脚本：

```bash
bash start.sh      # Linux/macOS
./start.command    # macOS 双击
start.bat          # Windows
```

---

🎭 自定义 AI 人设

1. 编辑 persona.txt，写入人设描述，例如：
   ```
   你是一个幽默风趣的助手，喜欢用 emoji，回答时偶尔讲冷笑话。
   ```
2. 运行 python persona_chat.py 即可。

---

📚 私有知识库（RAG）

1. 在 Web 界面中上传 PDF 或 TXT 文档。
2. AI 会自动将文档分块、向量化并存入数据库。
3. 提问时，AI 会检索相关文档片段并结合生成回答。

---

🧪 模型量化

```bash
python quantize.py
```

生成 INT8 量化模型 checkpoints/quantized_model.pt，体积减少约 70%，推理速度提升 30-50%。

---

📱 移动端部署（Termux + Debian）

1. 安装 Termux，执行：
   ```bash
   pkg install proot-distro
   proot-distro install debian
   proot-distro login debian
   ```
2. 在 Debian 中安装 Python 和依赖，运行项目。

详见项目文档中的完整移动端部署指南。

---

🧠 知识蒸馏

项目支持使用教师模型生成的问答对进行蒸馏训练。将蒸馏数据（如 distillation_data.txt）放入项目目录，在 config.py 中配置即可。

---

📄 许可证

本项目采用 MIT 许可证，欢迎自由使用、修改和分发。

---