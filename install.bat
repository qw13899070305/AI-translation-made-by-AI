@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo    🤖 我的专属 AI 助手 环境安装 (Windows)
echo ========================================

:: 1. 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [√] Python 已安装

:: 2. 创建虚拟环境
if not exist venv (
    echo [√] 创建虚拟环境...
    python -m venv venv
)
echo [√] 虚拟环境已就绪

:: 3. 激活虚拟环境
call venv\Scripts\activate.bat

:: 4. 升级 pip
python -m pip install --upgrade pip -q

:: 5. 安装依赖
echo [√] 安装 Python 依赖...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo [√] 依赖安装完成

:: 6. 检查分词器
if not exist tokenizer\our_bpe.model (
    echo [!] 分词器未找到，正在训练...
    python tokenizer_train.py
    echo [√] 分词器训练完成
)

echo.
echo ========================================
echo    ✨ 环境安装完成！正在启动...
echo ========================================
echo.

:: 7. 调用已有的启动脚本
if exist start.ps1 (
    powershell -ExecutionPolicy Bypass -File start.ps1
) else if exist start.bat (
    call start.bat
) else (
    python chat_web.py
)

pause