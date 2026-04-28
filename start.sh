#!/bin/bash
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
cd "$(dirname "$0")"

# ---------- 语言选择 ----------
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}   🤖 My Own AI Assistant / 我的专属 AI 助手${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  Please select language / 请选择语言:"
echo -e "  1) English"
echo -e "  2) 中文"
read -p "  Enter choice (1 or 2): " lang_choice

if [ "$lang_choice" == "1" ]; then
    MSG_PYTHON_READY="✅ Python3 is ready."
    MSG_PYTHON_FAIL="❌ Python3 not found. Please install Python 3.8+"
    MSG_VENV_ACTIVATED="✅ Virtual environment activated."
    MSG_VENV_NOT_FOUND="⚠️  Virtual environment not found, using system Python."
    MSG_PYTORCH_OK="✅ PyTorch installed (version"
    MSG_PYTORCH_FAIL="❌ PyTorch not installed. Please run: pip install -r requirements.txt"
    MSG_TOKENIZER_OK="✅ Tokenizer exists."
    MSG_TOKENIZER_MISSING="⚠️  Tokenizer not found, training..."
    MSG_TOKENIZER_DONE="✅ Tokenizer training completed."
    MSG_NO_CHECKPOINT="⚠️  No model checkpoints found."
    MSG_SUGGEST_CONFIG="💡 Suggestion: Choose 0 to enter Configuration Center, then choose 6 to train."
    MSG_MAIN_MENU="Please select a startup mode:"
    MSG_MENU_CONFIG="🛠️  Configuration Center (Recommended before training)"
    MSG_MENU_WEB="🚀 Launch Web Interface"
    MSG_MENU_CLI="💬 Launch Command Line Chat"
    MSG_MENU_API="🌐 Launch API Backend"
    MSG_MENU_RAG="📚 Test RAG Module"
    MSG_MENU_PERSONA="🎭 Launch Persona Chat"
    MSG_MENU_TRAIN="📦 Train Model"
    MSG_MENU_TOKENIZER="🔧 Train Tokenizer"
    MSG_MENU_TOOLS="🧰 Standalone Tools"
    MSG_MENU_EXIT="👋 Exit"
    MSG_TOOLS_MENU="🧰 Standalone Tools Menu"
    MSG_TOOL1="Web Search"
    MSG_TOOL2="Export Conversation History"
    MSG_TOOL3="Model Quantization"
    MSG_TOOL4="Dataset Preview"
    MSG_TOOL5="Long-term Memory Test"
    MSG_TOOL6="Data Expansion (Download multi-source data)"
    MSG_TOOL7="Multi-task Learning Training"
    MSG_TOOL8="Continual Learning Training"
    MSG_TOOL9="🎓 MOPD Multi-Teacher Distillation"
    MSG_TOOL0="Return to Main Menu"
    MSG_INVALID="Invalid choice, please try again."
    MSG_GOODBYE="👋 Goodbye!"
    MSG_CONFIRM="⚠️  Checkpoint exists. Continue training will overwrite. Confirm? (y/n)"
    MSG_LAUNCH_CONFIG="🛠️  Launching Configuration Center..."
    MSG_LAUNCH_WEB="🚀 Launching Web Interface..."
    MSG_LAUNCH_CLI="💬 Launching Command Line Chat..."
    MSG_LAUNCH_API="🌐 Launching API Backend (port 8000)..."
    MSG_LAUNCH_RAG="📚 Testing RAG Module..."
    MSG_LAUNCH_PERSONA="🎭 Launching Persona Chat..."
    MSG_LAUNCH_TRAIN="📦 Starting model training..."
    MSG_LAUNCH_TOKENIZER="🔧 Training tokenizer..."
    MSG_BACKEND_MISSING="❌ backend directory or main.py not found"
else
    MSG_PYTHON_READY="✅ Python3 已就绪。"
    MSG_PYTHON_FAIL="❌ 未找到 python3，请先安装 Python 3.8+"
    MSG_VENV_ACTIVATED="✅ 已激活虚拟环境。"
    MSG_VENV_NOT_FOUND="⚠️  未找到虚拟环境，将使用系统 Python。"
    MSG_PYTORCH_OK="✅ PyTorch 已安装 (版本"
    MSG_PYTORCH_FAIL="❌ PyTorch 未安装，请运行: pip install -r requirements.txt"
    MSG_TOKENIZER_OK="✅ 分词器已存在。"
    MSG_TOKENIZER_MISSING="⚠️  分词器未找到，正在训练..."
    MSG_TOKENIZER_DONE="✅ 分词器训练完成。"
    MSG_NO_CHECKPOINT="⚠️  未找到任何模型检查点。"
    MSG_SUGGEST_CONFIG="💡 建议：选择 0 进入配置中心调整参数，然后选 6 训练模型。"
    MSG_MAIN_MENU="请选择启动模式:"
    MSG_MENU_CONFIG="🛠️  配置管理中心（建议训练前先配置）"
    MSG_MENU_WEB="🚀 启动 Web 界面"
    MSG_MENU_CLI="💬 启动命令行对话"
    MSG_MENU_API="🌐 启动 API 后端服务"
    MSG_MENU_RAG="📚 测试 RAG 模块"
    MSG_MENU_PERSONA="🎭 启动人设对话模式"
    MSG_MENU_TRAIN="📦 训练模型"
    MSG_MENU_TOKENIZER="🔧 训练分词器"
    MSG_MENU_TOOLS="🧰 独立工具"
    MSG_MENU_EXIT="👋 退出"
    MSG_TOOLS_MENU="🧰 独立工具菜单"
    MSG_TOOL1="联网搜索"
    MSG_TOOL2="对话历史导出"
    MSG_TOOL3="模型量化"
    MSG_TOOL4="数据集预览"
    MSG_TOOL5="长期记忆测试"
    MSG_TOOL6="数据扩充（下载多源数据）"
    MSG_TOOL7="多任务学习训练"
    MSG_TOOL8="持续学习训练"
    MSG_TOOL9="🎓 MOPD 多教师蒸馏"
    MSG_TOOL0="返回主菜单"
    MSG_INVALID="无效输入，请重新选择。"
    MSG_GOODBYE="👋 再见！"
    MSG_CONFIRM="⚠️  已有模型检查点，继续训练将覆盖旧模型。确认？(y/n)"
    MSG_LAUNCH_CONFIG="🛠️  正在启动配置管理中心..."
    MSG_LAUNCH_WEB="🚀 正在启动 Web 界面..."
    MSG_LAUNCH_CLI="💬 正在启动命令行对话..."
    MSG_LAUNCH_API="🌐 正在启动 API 后端 (端口 8000)..."
    MSG_LAUNCH_RAG="📚 正在测试 RAG 模块..."
    MSG_LAUNCH_PERSONA="🎭 正在启动人设对话模式..."
    MSG_LAUNCH_TRAIN="📦 正在开始训练模型..."
    MSG_LAUNCH_TOKENIZER="🔧 正在训练分词器..."
    MSG_BACKEND_MISSING="❌ backend 目录或 main.py 不存在"
fi

# ---------- 环境检查 ----------
check_python() {
    if command -v python3 &>/dev/null; then echo -e "${GREEN}$MSG_PYTHON_READY${NC}"
    else echo -e "${RED}$MSG_PYTHON_FAIL${NC}"; exit 1; fi
}
activate_venv() {
    if [ -d "venv" ]; then source venv/bin/activate; echo -e "${GREEN}$MSG_VENV_ACTIVATED${NC}"
    elif [ -d "../venv" ]; then source ../venv/bin/activate; echo -e "${GREEN}$MSG_VENV_ACTIVATED${NC}"
    else echo -e "${YELLOW}$MSG_VENV_NOT_FOUND${NC}"; fi
}
check_pytorch() {
    if python3 -c "import torch" 2>/dev/null; then
        local ver=$(python3 -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}$MSG_PYTORCH_OK $ver)${NC}"
    else echo -e "${RED}$MSG_PYTORCH_FAIL${NC}"; exit 1; fi
}
check_tokenizer() {
    if [ -f "tokenizer/our_bpe.model" ]; then echo -e "${GREEN}$MSG_TOKENIZER_OK${NC}"
    else
        echo -e "${YELLOW}$MSG_TOKENIZER_MISSING${NC}"
        python3 tokenizer_train.py
        echo -e "${GREEN}$MSG_TOKENIZER_DONE${NC}"
    fi
}
check_model() {
    local ckpt=$(find checkpoints -name "*.pt" 2>/dev/null | wc -l)
    local lora=$(find lora_weights -name "*.pt" 2>/dev/null | wc -l)
    if [ $ckpt -gt 0 ] || [ $lora -gt 0 ]; then return 0; else return 1; fi
}

# ---------- 工具菜单 ----------
run_tools_menu() {
    while true; do
        echo ""
        echo -e "${BLUE}$MSG_TOOLS_MENU${NC}"
        echo "  1) $MSG_TOOL1"
        echo "  2) $MSG_TOOL2"
        echo "  3) $MSG_TOOL3"
        echo "  4) $MSG_TOOL4"
        echo "  5) $MSG_TOOL5"
        echo "  6) $MSG_TOOL6"
        echo "  7) $MSG_TOOL7"
        echo "  8) $MSG_TOOL8"
        echo "  9) $MSG_TOOL9"
        echo "  0) $MSG_TOOL0"
        echo ""
        read -p "  > " tool_choice
        case $tool_choice in
            1) python3 web_search.py ;;
            2) python3 export_history.py ;;
            3) python3 quantize.py ;;
            4) python3 preview_data.py ;;
            5) python3 recall.py ;;
            6) python3 enhanced_data_loader.py ;;
            7) python3 multitask_trainer.py ;;
            8) python3 continual_trainer.py ;;
            9) python3 distill_mopd.py ;;
            0) return ;;
            *) echo -e "${RED}$MSG_INVALID${NC}" ;;
        esac
        read -p "  Press Enter to continue..."
    done
}

# ---------- 主流程 ----------
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}   🤖 My Own AI Assistant / 我的专属 AI 助手${NC}"
echo -e "${BLUE}========================================${NC}"
check_python; activate_venv; check_pytorch; check_tokenizer

if ! check_model; then
    echo -e "${YELLOW}$MSG_NO_CHECKPOINT${NC}"
    echo -e "${YELLOW}$MSG_SUGGEST_CONFIG${NC}"
fi

while true; do
    echo ""
    echo -e "${BLUE}$MSG_MAIN_MENU${NC}"
    echo "  0) $MSG_MENU_CONFIG"
    echo "  1) $MSG_MENU_WEB"
    echo "  2) $MSG_MENU_CLI"
    echo "  3) $MSG_MENU_API"
    echo "  4) $MSG_MENU_RAG"
    echo "  5) $MSG_MENU_PERSONA"
    echo "  6) $MSG_MENU_TRAIN"
    echo "  7) $MSG_MENU_TOKENIZER"
    echo "  8) $MSG_MENU_TOOLS"
    echo "  9) $MSG_MENU_EXIT"
    echo ""
    read -p "  > " choice

    case $choice in
        0) echo -e "${GREEN}$MSG_LAUNCH_CONFIG${NC}"; python3 config_manager.py ;;
        1) echo -e "${GREEN}$MSG_LAUNCH_WEB${NC}"; python3 chat_web.py; break ;;
        2) echo -e "${GREEN}$MSG_LAUNCH_CLI${NC}"; python3 chat_cli.py; break ;;
        3) echo -e "${GREEN}$MSG_LAUNCH_API${NC}"; [ -d backend ] && [ -f backend/main.py ] && cd backend && python3 main.py || echo -e "${RED}$MSG_BACKEND_MISSING${NC}"; break ;;
        4) echo -e "${GREEN}$MSG_LAUNCH_RAG${NC}"; python3 -c "from rag_module import RAGModule; print('✅ RAG OK')"; break ;;
        5) echo -e "${GREEN}$MSG_LAUNCH_PERSONA${NC}"; python3 persona_chat.py; break ;;
        6)
            if check_model; then
                echo -e "${YELLOW}$MSG_CONFIRM${NC}"; read -r confirm
                [[ "$confirm" =~ ^[Yy]$ ]] && echo -e "${GREEN}$MSG_LAUNCH_TRAIN${NC}" && python3 train.py
            else
                echo -e "${GREEN}$MSG_LAUNCH_TRAIN${NC}"
                python3 train.py
            fi
            ;;
        7) echo -e "${GREEN}$MSG_LAUNCH_TOKENIZER${NC}"; python3 tokenizer_train.py ;;
        8) run_tools_menu ;;
        9) echo -e "${YELLOW}$MSG_GOODBYE${NC}"; exit 0 ;;
        *) echo -e "${RED}$MSG_INVALID${NC}" ;;
    esac
done