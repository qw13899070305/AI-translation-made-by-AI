#!/bin/bash
# ============================================
# 我的专属 AI 助手 - 一键安装 (macOS)
# 只安装缺失依赖，完成后调用 start.command
# ============================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}   🤖 我的专属 AI 助手 环境安装 (macOS)${NC}"
echo -e "${BLUE}========================================${NC}"

# 1. 检查 Homebrew（macOS 包管理器）
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}⚠️  未检测到 Homebrew${NC}"
    echo -e "${YELLOW}   Homebrew 是 macOS 上最常用的包管理器，用于安装 Python 等工具。${NC}"
    echo -e "${YELLOW}   是否立即安装 Homebrew？(y/n)${NC}"
    read -r install_brew
    if [[ "$install_brew" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}📥 正在安装 Homebrew...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # 将 Homebrew 添加到 PATH（Apple Silicon 和 Intel 路径不同）
        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
        echo -e "${GREEN}✅ Homebrew 安装完成${NC}"
    else
        echo -e "${RED}❌ Homebrew 是必需的工具，安装中止。${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Homebrew 已安装${NC}"
fi

# 2. 检查 Python 版本（macOS 预装的可能较旧）
PYTHON_OK=false
if command -v python3 &> /dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 8 ]; then
        PYTHON_OK=true
        echo -e "${GREEN}✅ Python $PY_VER 已安装（版本符合要求）${NC}"
    else
        echo -e "${YELLOW}⚠️  Python 版本 $PY_VER 过低（需要 3.8+）${NC}"
    fi
fi

if [ "$PYTHON_OK" = false ]; then
    echo -e "${BLUE}📥 通过 Homebrew 安装 Python 3.12...${NC}"
    brew install python@3.12
    # 链接到 python3 命令
    brew link --overwrite python@3.12
    echo -e "${GREEN}✅ Python 安装完成${NC}"
fi

# 3. 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo -e "${BLUE}📦 创建虚拟环境...${NC}"
    python3 -m venv venv
fi
echo -e "${GREEN}✅ 虚拟环境已就绪${NC}"

# 4. 激活虚拟环境并升级 pip
source venv/bin/activate
pip install --upgrade pip -q

# 5. 安装 Python 依赖
echo -e "${BLUE}📥 安装 Python 依赖...${NC}"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo -e "${GREEN}✅ 依赖安装完成${NC}"

# 6. 检查分词器，若无则训练
if [ ! -f "tokenizer/our_bpe.model" ]; then
    echo -e "${YELLOW}🔧 分词器未找到，正在训练...${NC}"
    python tokenizer_train.py
    echo -e "${GREEN}✅ 分词器训练完成${NC}"
fi

echo ""
echo -e "${GREEN}✨ 环境安装完成！正在启动...${NC}"
echo ""

# 7. 调用启动脚本（优先 start.command，其次 start.sh）
if [ -f "start.command" ]; then
    bash start.command
elif [ -f "start.sh" ]; then
    bash start.sh
else
    echo -e "${YELLOW}⚠️ 未找到启动脚本，直接启动 Web 界面...${NC}"
    python chat_web.py
fi