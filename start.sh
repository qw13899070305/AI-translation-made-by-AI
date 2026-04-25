#!/bin/bash
set -e

# ============================================
# 颜色定义
# ============================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

cd "$(dirname "$0")"

# ============================================
# 环境检查函数
# ============================================

check_python() {
    if ! command -v python3 &>/dev/null; then
        echo -e "${RED}❌ 未找到 python3，请先安装 Python 3.8+${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Python3 已就绪${NC}"
}

activate_venv() {
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo -e "${GREEN}✅ 已激活本地虚拟环境${NC}"
    elif [ -d "../venv" ]; then
        source ../venv/bin/activate
        echo -e "${GREEN}✅ 已激活上层虚拟环境${NC}"
    else
        echo -e "${YELLOW}⚠️  未找到虚拟环境，将使用系统 Python${NC}"
    fi
}

check_pytorch() {
    if python3 -c "import torch" 2>/dev/null; then
        echo -e "${GREEN}✅ PyTorch 已安装 (版本 $(python3 -c "import torch; print(torch.__version__)"))${NC}"
    else
        echo -e "${RED}❌ PyTorch 未安装，请运行: pip install -r requirements.txt${NC}"
        exit 1
    fi
}

check_tokenizer() {
    if [ -f "tokenizer/our_bpe.model" ]; then
        echo -e "${GREEN}✅ 分词器已存在${NC}"
    else
        echo -e "${YELLOW}⚠️  分词器缺失，开始自动训练...${NC}"
        python3 tokenizer_train.py
        echo -e "${GREEN}✅ 分词器训练完成${NC}"
    fi
}

check_model() {
    local ckpt_count=$(find checkpoints -name "*.pt" 2>/dev/null | wc -l)
    local lora_count=$(find lora_weights -name "*.pt" 2>/dev/null | wc -l)
    if [ $ckpt_count -gt 0 ] || [ $lora_count -gt 0 ]; then
        return 0
    fi
    return 1
}

# ============================================
# 主菜单
# ============================================
show_main_menu() {
    echo ""
    echo -e "${BLUE}请选择启动模式:${NC}"
    echo "  0) 🛠️  配置管理中心（建议训练前先配置）"
    echo "  1) 🚀 启动 Web 界面"
    echo "  2) 💬 启动命令行对话"
    echo "  3) 🌐 启动 API 后端服务"
    echo "  4) 📚 测试 RAG 模块"
    echo "  5) 🎭 启动人设对话模式"
    echo "  6) 📦 训练模型"
    echo "  7) 🔧 训练分词器"
    echo "  8) 🧰 独立工具"
    echo "  9) 👋 退出"
    echo ""
}

run_tools_menu() {
    while true; do
        echo ""
        echo -e "${BLUE}🧰 独立工具菜单${NC}"
        echo "  1) 联网搜索"
        echo "  2) 对话历史导出"
        echo "  3) 模型量化"
        echo "  4) 数据集预览"
        echo "  5) 长期记忆测试"
        echo "  6) 数据扩充（下载多源数据）"
        echo "  7) 多任务学习训练"
        echo "  8) 持续学习训练"
        echo "  9) 返回主菜单"
        echo ""
        read -p "请输入数字 (1-9): " tool_choice

        case $tool_choice in
            1) python3 web_search.py ;;
            2) python3 export_history.py ;;
            3) python3 quantize.py ;;
            4) python3 preview_data.py ;;
            5) python3 recall.py ;;
            6) python3 enhanced_data_loader.py ;;
            7) python3 multitask_trainer.py ;;
            8) python3 continual_trainer.py ;;
            9) return ;;
            *) echo -e "${RED}无效输入，请重新选择${NC}" ;;
        esac
        read -p "按回车键继续..."
    done
}

# ============================================
# 主流程
# ============================================
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}   🤖 我的专属 AI 助手 启动器${NC}"
    echo -e "${BLUE}========================================${NC}"

    check_python
    activate_venv
    check_pytorch
    check_tokenizer

    # 引导建议：如果没有模型检查点，提示用户先配置后训练
    if ! check_model; then
        echo -e "${YELLOW}⚠️  未找到任何模型检查点${NC}"
        echo -e "${YELLOW}💡 建议：选择 0 进入配置中心调整参数，然后选 6 训练模型${NC}"
    fi

    while true; do
        show_main_menu
        read -p "请输入数字 (0-9): " choice

        case $choice in
            0)
                echo -e "${GREEN}🛠️  启动配置管理中心...${NC}"
                python3 config_manager.py
                ;;
            1)
                echo -e "${GREEN}🚀 正在启动 Web 界面...${NC}"
                python3 chat_web.py
                break
                ;;
            2)
                echo -e "${GREEN}💬 正在启动命令行对话...${NC}"
                python3 chat_cli.py
                break
                ;;
            3)
                echo -e "${GREEN}🌐 正在启动 API 后端 (端口 8000)...${NC}"
                if [ -d "backend" ] && [ -f "backend/main.py" ]; then
                    cd backend
                    python3 main.py
                else
                    echo -e "${RED}❌ backend 目录或 main.py 不存在${NC}"
                fi
                break
                ;;
            4)
                echo -e "${GREEN}📚 测试 RAG 模块...${NC}"
                python3 -c "from rag_module import RAGModule; r = RAGModule(); print('✅ RAG 模块加载成功')"
                break
                ;;
            5)
                echo -e "${GREEN}🎭 启动人设对话模式...${NC}"
                python3 persona_chat.py
                break
                ;;
            6)
                if check_model; then
                    echo -e "${YELLOW}⚠️  已有模型检查点，继续训练将覆盖旧模型。确认？(y/n)${NC}"
                    read -r confirm
                    if [[ "$confirm" =~ ^[Yy]$ ]]; then
                        python3 train.py
                    fi
                else
                    echo -e "${GREEN}📦 开始训练模型...${NC}"
                    python3 train.py
                fi
                ;;
            7)
                echo -e "${GREEN}🔧 训练分词器...${NC}"
                python3 tokenizer_train.py
                ;;
            8)
                run_tools_menu
                ;;
            9)
                echo -e "${YELLOW}👋 再见！${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效输入，请重新选择${NC}"
                ;;
        esac
    done
}

main