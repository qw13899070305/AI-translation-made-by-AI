#!/bin/bash
# ============================================
# 我的专属 AI 助手 - 一键安装 (Linux 全发行版·中英双语)
# ============================================

set -e

# ---------- 语言选择 ----------
echo "========================================"
echo "   🤖 My Own AI Assistant Installer"
echo "========================================"
echo "  Please select language / 请选择语言:"
echo "  1) English"
echo "  2) 中文"
echo ""
read -p "Enter choice (1 or 2): " lang_choice

if [ "$lang_choice" == "1" ]; then
    MSG_WELCOME="Starting installation..."
    MSG_DETECT="🔍 Detecting system environment..."
    MSG_FOUND="✅ Detected system:"
    MSG_NOT_FOUND="❌ Unable to identify your Linux distribution."
    MSG_INSTALL_DEPS="📦 Installing system dependencies (sudo may be required)..."
    MSG_PYTHON_OK="✅ Python3 is already installed."
    MSG_PYTHON_INSTALL="⚠️  Python3 not found, installing via package manager..."
    MSG_VENV="📦 Creating Python virtual environment..."
    MSG_VENV_OK="✅ Virtual environment is ready."
    MSG_PIP_INSTALL="📥 Installing Python dependencies..."
    MSG_PIP_OK="✅ Python dependencies installed."
    MSG_TOKENIZER="🔧 Tokenizer not found, training..."
    MSG_TOKENIZER_OK="✅ Tokenizer training completed."
    MSG_DONE="✨ Environment setup complete! Launching AI..."
    MSG_START_NOT_FOUND="⚠️  start.sh not found, launching Web UI directly..."
    MSG_UNSUPPORTED_PM="❌ Unsupported package manager. Please install Python manually."
else
    MSG_WELCOME="开始安装..."
    MSG_DETECT="🔍 正在检测系统环境..."
    MSG_FOUND="✅ 检测到系统:"
    MSG_NOT_FOUND="❌ 无法识别你的 Linux 发行版。"
    MSG_INSTALL_DEPS="📦 正在安装系统依赖 (可能需要sudo权限)..."
    MSG_PYTHON_OK="✅ Python3 已安装。"
    MSG_PYTHON_INSTALL="⚠️  未找到 Python3，正在通过包管理器安装..."
    MSG_VENV="📦 正在创建 Python 虚拟环境..."
    MSG_VENV_OK="✅ 虚拟环境已就绪。"
    MSG_PIP_INSTALL="📥 正在安装 Python 依赖..."
    MSG_PIP_OK="✅ Python 依赖安装完成。"
    MSG_TOKENIZER="🔧 分词器未找到，正在训练..."
    MSG_TOKENIZER_OK="✅ 分词器训练完成。"
    MSG_DONE="✨ 环境安装完成！正在启动 AI..."
    MSG_START_NOT_FOUND="⚠️  未找到 start.sh，将直接启动 Web 界面..."
    MSG_UNSUPPORTED_PM="❌ 不支持的包管理器，请手动安装 Python。"
fi

echo -e "\n$MSG_WELCOME\n"

# ---------- 智能探测 Linux 发行版 ----------
echo -e "$MSG_DETECT"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    echo -e "$MSG_NOT_FOUND"
    exit 1
fi

PYTHON_DEV_PKG=""
case $OS in
    ubuntu|debian)
        PKG_MANAGER="apt-get"
        PYTHON_DEV_PKG="python3-venv python3-dev"
        ;;
    centos|rhel|fedora|rocky|almalinux)
        PKG_MANAGER="dnf"
        PYTHON_DEV_PKG="python3-devel"
        if [ "$OS" = "centos" ] && [ "${OS_VERSION%%.*}" -le 7 ]; then
            PKG_MANAGER="yum"
        fi
        ;;
    arch|manjaro)
        PKG_MANAGER="pacman"
        PYTHON_DEV_PKG="base-devel"
        ;;
    alpine)
        PKG_MANAGER="apk"
        PYTHON_DEV_PKG="python3-dev"
        ;;
    opensuse*|sles)
        PKG_MANAGER="zypper"
        PYTHON_DEV_PKG="python3-devel"
        ;;
    *)
        PKG_MANAGER="apt-get"
        PYTHON_DEV_PKG="python3-venv python3-dev"
        ;;
esac

echo -e "$MSG_FOUND $OS, using $PKG_MANAGER"

# ---------- 安装系统依赖 ----------
echo -e "$MSG_INSTALL_DEPS"
if ! command -v python3 &> /dev/null; then
    echo -e "$MSG_PYTHON_INSTALL"
    case $PKG_MANAGER in
        apt-get) sudo apt-get update && sudo apt-get install -y python3 python3-pip $PYTHON_DEV_PKG ;;
        dnf) sudo dnf install -y python3 python3-pip $PYTHON_DEV_PKG ;;
        yum) sudo yum install -y python3 python3-pip $PYTHON_DEV_PKG ;;
        pacman) sudo pacman -Sy --noconfirm python python-pip $PYTHON_DEV_PKG ;;
        apk) sudo apk add --no-cache python3 py3-pip $PYTHON_DEV_PKG ;;
        zypper) sudo zypper --non-interactive install python3 python3-pip $PYTHON_DEV_PKG ;;
        *) echo -e "$MSG_UNSUPPORTED_PM" && exit 1 ;;
    esac
else
    echo -e "$MSG_PYTHON_OK"
fi

# ---------- 创建虚拟环境 ----------
echo -e "$MSG_VENV"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
echo -e "$MSG_VENV_OK"

# ---------- 安装 Python 依赖 ----------
source venv/bin/activate
pip install --upgrade pip -q
echo -e "$MSG_PIP_INSTALL"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo -e "$MSG_PIP_OK"

# ---------- 训练分词器 ----------
if [ ! -f "tokenizer/our_bpe.model" ]; then
    echo -e "$MSG_TOKENIZER"
    python tokenizer_train.py
    echo -e "$MSG_TOKENIZER_OK"
fi

# ---------- 启动 ----------
echo -e "$MSG_DONE"
if [ -f "start.sh" ]; then
    bash start.sh
else
    echo -e "$MSG_START_NOT_FOUND"
    python chat_web.py
fi