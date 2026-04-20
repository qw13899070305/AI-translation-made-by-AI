#!/bin/bash

# ============================================
# 我的专属 AI 助手 - 启动脚本
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}   🤖 我的专属 AI 助手 启动器${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误：未找到 python3，请先安装 Python 3.8+${NC}"
    exit 1
fi

# 检查并激活虚拟环境
if [ -d "venv" ]; then
    echo -e "${GREEN}✅ 发现虚拟环境，正在激活...${NC}"
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo -e "${GREEN}✅ 发现虚拟环境，正在激活...${NC}"
    source ../venv/bin/activate
else
    echo -e "${YELLOW}⚠️  未找到虚拟环境，使用系统 Python${NC}"
fi

# 检查依赖是否安装
echo -e "${BLUE}🔍 检查依赖...${NC}"
python3 -c "import torch; print(f'PyTorch 版本: {torch.__version__}')" 2>/dev/null || {
    echo -e "${RED}错误：PyTorch 未安装，请先运行: pip install -r requirements.txt${NC}"
    exit 1
}

# 检查分词器是否存在
if [ ! -f "tokenizer/our_bpe.model" ]; then
    echo -e "${YELLOW}⚠️  分词器未找到，正在自动训练...${NC}"
    python3 tokenizer_train.py
fi

# 检查模型检查点
CKPT_COUNT=$(find checkpoints -name "*.pt" 2>/dev/null | wc -l)
LORA_COUNT=$(find lora_weights -name "*.pt" 2>/dev/null | wc -l)
if [ $CKPT_COUNT -eq 0 ] && [ $LORA_COUNT -eq 0 ]; then
    echo -e "${YELLOW}⚠️  未找到任何模型检查点，您需要先训练模型。${NC}"
    echo -e "${YELLOW}   是否立即开始训练？(y/n)${NC}"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        python3 train.py
    else
        echo -e "${YELLOW}   将使用随机初始化模型（效果可能不佳）。${NC}"
    fi
fi

# 显示菜单
echo ""
echo -e "${BLUE}请选择启动模式:${NC}"
echo "  1) Web 界面 (Gradio) - 推荐，支持图文和 RAG"
echo "  2) 命令行对话 (CLI)"
echo "  3) API 后端服务 (FastAPI) - 供手机端调用"
echo "  4) 仅启动 RAG 向量数据库测试"
echo "  5) 退出"
echo ""
read -p "输入数字 (1-5): " choice

case $choice in
    1)
        echo -e "${GREEN}🚀 启动 Web 界面...${NC}"
        python3 chat_web.py
        ;;
    2)
        echo -e "${GREEN}💬 启动命令行对话...${NC}"
        python3 chat_cli.py
        ;;
    3)
        echo -e "${GREEN}🌐 启动 API 后端服务 (端口 8000)...${NC}"
        cd backend 2>/dev/null || { echo -e "${RED}错误：backend 目录不存在${NC}"; exit 1; }
        python3 main.py
        ;;
    4)
        echo -e "${GREEN}📚 测试 RAG 模块...${NC}"
        python3 -c "from rag_module import RAGModule; r = RAGModule(); print('RAG 模块加载成功！')"
        ;;
    5)
        echo -e "${YELLOW}👋 再见！${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选择，退出。${NC}"
        exit 1
        ;;
esac