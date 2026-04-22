#!/bin/bash
# ============================================
# 我的专属 AI 助手 - 一键安装 (Linux 全发行版)
# ============================================

set -e

# 颜色定义 (略)
# ... (与之前版本保持一致，此处省略，确保有颜色变量) ...
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}   🤖 我的专属 AI 助手 环境安装 (Linux)${NC}"
echo -e "${BLUE}========================================${NC}"

# 1. 智能探测Linux发行版及包管理器
echo -e "${BLUE}🔍 正在检测系统环境...${NC}"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    echo -e "${RED}❌ 无法识别当前Linux发行版${NC}"
    exit 1
fi

# 定义系统依赖包名 (根据不同发行版进行映射)
PYTHON_DEV_PKG=""
case $OS in
    ubuntu|debian)
        PKG_MANAGER="apt-get"
        PYTHON_DEV_PKG="python3-venv python3-dev"
        ;;
    centos|rhel|fedora|rocky|almalinux)
        PKG_MANAGER="dnf"  # 现代RHEL系使用dnf
        PYTHON_DEV_PKG="python3-devel"
        # 如果是老版本CentOS 7，则使用yum
        if [ "$OS" = "centos" ] && [ "${OS_VERSION%%.*}" -le 7 ]; then
            PKG_MANAGER="yum"
        fi
        ;;
    arch|manjaro)
        PKG_MANAGER="pacman"
        PYTHON_DEV_PKG="base-devel" # Arch Linux的编译工具链
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
        echo -e "${YELLOW}⚠️ 未知发行版: $OS，将尝试使用apt-get${NC}"
        PKG_MANAGER="apt-get"
        PYTHON_DEV_PKG="python3-venv python3-dev"
        ;;
esac

echo -e "${GREEN}✅ 检测到系统: $OS, 将使用包管理器: $PKG_MANAGER${NC}"

# 2. 检查并安装Python和系统级依赖
echo -e "${BLUE}📦 检查并安装系统级依赖 (可能需要sudo权限)...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}⚠️ 未找到 python3，正在通过 $PKG_MANAGER 安装...${NC}"
    case $PKG_MANAGER in
        apt-get) sudo apt-get update && sudo apt-get install -y python3 python3-pip $PYTHON_DEV_PKG ;;
        dnf) sudo dnf install -y python3 python3-pip $PYTHON_DEV_PKG ;;
        yum) sudo yum install -y python3 python3-pip $PYTHON_DEV_PKG ;;
        pacman) sudo pacman -Sy --noconfirm python python-pip $PYTHON_DEV_PKG ;;
        apk) sudo apk add --no-cache python3 py3-pip $PYTHON_DEV_PKG ;;
        zypper) sudo zypper --non-interactive install python3 python3-pip $PYTHON_DEV_PKG ;;
        *) echo -e "${RED}❌ 不支持的包管理器，请手动安装Python。${NC}" && exit 1 ;;
    esac
else
    echo -e "${GREEN}✅ Python3 已安装${NC}"
fi

# 3. 创建虚拟环境 (后续步骤与之前相同，但移除了用户交互，直接执行)
# ... (后续内容：创建venv、安装pip依赖、训练分词器) ...

# 注意：在最后一步，直接调用启动脚本 start.sh，不再显示用户菜单。
if [ -f "start.sh" ]; then
    echo -e "${GREEN}✨ 环境安装完成！正在启动...${NC}"
    bash start.sh
else
    echo -e "${YELLOW}⚠️ 未找到 start.sh，将直接启动 Web 界面...${NC}"
    python chat_web.py
fi