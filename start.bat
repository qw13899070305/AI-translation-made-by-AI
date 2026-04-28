@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo    🤖 My Own AI Assistant / 我的专属 AI 助手
echo ========================================
echo   Please select language / 请选择语言:
echo   1) English
echo   2) 中文
set /p lang_choice="Enter choice (1 or 2): "

if "%lang_choice%"=="1" (
    set MSG_PYTHON_FAIL=[Error] Python not found. Please install Python 3.8+
    set MSG_VENV_ACTIVATE=Activating virtual environment...
    set MSG_VENV_NOT_FOUND=⚠️  Virtual environment not found, using system Python.
    set MSG_PYTORCH_FAIL=❌ PyTorch not installed. Please run: pip install -r requirements.txt
    set MSG_TOKENIZER_TRAIN=⚠️  Tokenizer not found, training...
    set MSG_MAIN_MENU=Please select a startup mode:
    set MSG_MENU_CONFIG=🛠️  Configuration Center
    set MSG_MENU_WEB=🚀 Launch Web Interface (text/image/voice/doc)
    set MSG_MENU_CLI=💬 Launch Command Line Chat
    set MSG_MENU_API=🌐 Launch API Backend (port 8000)
    set MSG_MENU_RAG=📚 Test RAG Knowledge Base
    set MSG_MENU_PERSONA=🎭 Launch Persona Chat
    set MSG_MENU_TRAIN=📦 Train Model (supports Muon/AMP)
    set MSG_MENU_TOKENIZER=🔧 Train Tokenizer
    set MSG_MENU_TOOLS=🧰 Standalone Tools
    set MSG_MENU_EXIT=👋 Exit
    set MSG_TOOL1=Web Search
    set MSG_TOOL2=Export Conversation History
    set MSG_TOOL3=Model Quantization
    set MSG_TOOL4=Dataset Preview
    set MSG_TOOL5=Long-term Memory Test
    set MSG_TOOL6=Data Expansion
    set MSG_TOOL7=Multi-task Learning
    set MSG_TOOL8=Continual Learning
    set MSG_TOOL9=🎓 MOPD Multi-Teacher Distillation
    set MSG_TOOL0=Return to Main Menu
    set MSG_GOODBYE=👋 Goodbye!
) else (
    set MSG_PYTHON_FAIL=[错误] 未找到 Python，请先安装 Python 3.8+
    set MSG_VENV_ACTIVATE=正在激活虚拟环境...
    set MSG_VENV_NOT_FOUND=⚠️  未找到虚拟环境，使用系统 Python。
    set MSG_PYTORCH_FAIL=❌ PyTorch 未安装，请运行: pip install -r requirements.txt
    set MSG_TOKENIZER_TRAIN=⚠️  分词器未找到，正在训练...
    set MSG_MAIN_MENU=请选择启动模式:
    set MSG_MENU_CONFIG=🛠️  配置管理中心
    set MSG_MENU_WEB=🚀 启动 Web 界面（支持图文/语音/文档上传）
    set MSG_MENU_CLI=💬 启动命令行对话
    set MSG_MENU_API=🌐 启动 API 后端服务 (端口 8000)
    set MSG_MENU_RAG=📚 测试 RAG 知识库模块
    set MSG_MENU_PERSONA=🎭 启动人设对话模式
    set MSG_MENU_TRAIN=📦 训练模型（支持 Muon/AMP）
    set MSG_MENU_TOKENIZER=🔧 训练分词器
    set MSG_MENU_TOOLS=🧰 独立工具
    set MSG_MENU_EXIT=👋 退出
    set MSG_TOOL1=联网搜索
    set MSG_TOOL2=对话历史导出
    set MSG_TOOL3=模型量化
    set MSG_TOOL4=数据集预览
    set MSG_TOOL5=长期记忆测试
    set MSG_TOOL6=数据扩充
    set MSG_TOOL7=多任务学习
    set MSG_TOOL8=持续学习
    set MSG_TOOL9=🎓 MOPD 多教师蒸馏
    set MSG_TOOL0=返回主菜单
    set MSG_GOODBYE=👋 再见！
)

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo %MSG_PYTHON_FAIL%
    pause
    exit /b 1
)
:: Activate venv
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo %MSG_VENV_NOT_FOUND%
)
:: Check PyTorch
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo %MSG_PYTORCH_FAIL%
    pause
    exit /b 1
)
:: Check Tokenizer
if not exist tokenizer\our_bpe.model (
    echo %MSG_TOKENIZER_TRAIN%
    python tokenizer_train.py
)

:main_menu
echo.
echo %MSG_MAIN_MENU%
echo   0) %MSG_MENU_CONFIG%
echo   1) %MSG_MENU_WEB%
echo   2) %MSG_MENU_CLI%
echo   3) %MSG_MENU_API%
echo   4) %MSG_MENU_RAG%
echo   5) %MSG_MENU_PERSONA%
echo   6) %MSG_MENU_TRAIN%
echo   7) %MSG_MENU_TOKENIZER%
echo   8) %MSG_MENU_TOOLS%
echo   9) %MSG_MENU_EXIT%
set /p choice="Enter choice (0-9): "

if "%choice%"=="0" ( python config_manager.py && goto main_menu )
if "%choice%"=="1" ( python chat_web.py && goto end )
if "%choice%"=="2" ( python chat_cli.py && goto end )
if "%choice%"=="3" ( if exist backend\main.py ( cd backend && python main.py && cd .. ) else ( echo backend not found ) && goto end )
if "%choice%"=="4" ( python -c "from rag_module import RAGModule; print('RAG OK')" && goto main_menu )
if "%choice%"=="5" ( python persona_chat.py && goto end )
if "%choice%"=="6" ( python train.py && goto main_menu )
if "%choice%"=="7" ( python tokenizer_train.py && goto main_menu )
if "%choice%"=="8" goto tools_menu
if "%choice%"=="9" ( echo %MSG_GOODBYE% && goto end )
echo Invalid choice
goto main_menu

:tools_menu
echo.
echo %MSG_MENU_TOOLS%
echo   1) %MSG_TOOL1%
echo   2) %MSG_TOOL2%
echo   3) %MSG_TOOL3%
echo   4) %MSG_TOOL4%
echo   5) %MSG_TOOL5%
echo   6) %MSG_TOOL6%
echo   7) %MSG_TOOL7%
echo   8) %MSG_TOOL8%
echo   9) %MSG_TOOL9%
echo   0) %MSG_TOOL0%
set /p tchoice="> "
if "%tchoice%"=="1" python web_search.py
if "%tchoice%"=="2" python export_history.py
if "%tchoice%"=="3" python quantize.py
if "%tchoice%"=="4" python preview_data.py
if "%tchoice%"=="5" python recall.py
if "%tchoice%"=="6" python enhanced_data_loader.py
if "%tchoice%"=="7" python multitask_trainer.py
if "%tchoice%"=="8" python continual_trainer.py
if "%tchoice%"=="9" python distill_mopd.py
if "%tchoice%"=="0" goto main_menu
pause
goto tools_menu

:end
pause