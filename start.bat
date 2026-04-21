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

echo 启动 Web 界面...
python chat_web.py
pause