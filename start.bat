@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo    🤖 我的专属 AI 助手 启动器 (Windows)
echo ========================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 3.8+
    pause
    exit /b 1
)

if exist venv\Scripts\activate.bat (
    echo [√] 激活虚拟环境...
    call venv\Scripts\activate.bat
)

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [错误] PyTorch 未安装，请先运行: pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist tokenizer\our_bpe.model (
    echo [!] 训练分词器...
    python tokenizer_train.py
)

:menu
echo.
echo 请选择启动模式:
echo   1) Web 界面 (Gradio)
echo   2) 命令行对话 (CLI)
echo   3) API 后端服务
echo   4) 测试 RAG 模块
echo   5) 人设对话模式 (Persona Chat)
echo   6) 退出
set /p choice="输入数字 (1-6): "

if "%choice%"=="1" goto web
if "%choice%"=="2" goto cli
if "%choice%"=="3" goto api
if "%choice%"=="4" goto rag
if "%choice%"=="5" goto persona
if "%choice%"=="6" goto end
echo 无效选择，请重新输入。
goto menu

:web
echo 启动 Web 界面...
python chat_web.py
pause
goto end

:cli
echo 启动命令行对话...
python chat_cli.py
pause
goto end

:api
echo 启动 API 后端服务 (端口 8000)...
if exist backend\main.py (
    cd backend
    python main.py
    cd ..
) else (
    echo [错误] backend 目录不存在
)
pause
goto end

:rag
echo 测试 RAG 模块...
python -c "from rag_module import RAGModule; r = RAGModule(); print('RAG 模块加载成功！')"
pause
goto end

:persona
echo 启动人设对话模式...
python persona_chat.py
pause
goto end

:end
echo 再见！