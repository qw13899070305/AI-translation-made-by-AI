@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo    🤖 My Own AI Assistant / 我的专属 AI 助手
echo ========================================
echo   Please select language / 请选择语言:
echo   1) English
echo   2) 中文
echo.
set /p lang_choice="Enter choice (1 or 2): "

if "%lang_choice%"=="1" (
    set MSG_PYTHON_CHECK=Checking Python...
    set MSG_PYTHON_FAIL=[Error] Python not found. Please install Python 3.8+
    set MSG_PYTHON_DOWNLOAD=Download: https://www.python.org/downloads/
    set MSG_VENV_CREATE=Creating virtual environment...
    set MSG_VENV_READY=Virtual environment is ready.
    set MSG_VENV_ACTIVATE=Activating virtual environment...
    set MSG_PIP_UPGRADE=Upgrading pip...
    set MSG_PIP_INSTALL=Installing Python dependencies...
    set MSG_PIP_DONE=Dependencies installed.
    set MSG_TOKENIZER_CHECK=Checking tokenizer...
    set MSG_TOKENIZER_TRAIN=Tokenizer not found, training...
    set MSG_TOKENIZER_DONE=Tokenizer training completed.
    set MSG_ENV_DONE=Environment setup complete! Launching...
    set MSG_PRESS_KEY=Press any key to exit...
) else (
    set MSG_PYTHON_CHECK=正在检查 Python...
    set MSG_PYTHON_FAIL=[错误] 未找到 Python，请先安装 Python 3.8+
    set MSG_PYTHON_DOWNLOAD=下载地址: https://www.python.org/downloads/
    set MSG_VENV_CREATE=正在创建虚拟环境...
    set MSG_VENV_READY=虚拟环境已就绪。
    set MSG_VENV_ACTIVATE=正在激活虚拟环境...
    set MSG_PIP_UPGRADE=正在升级 pip...
    set MSG_PIP_INSTALL=正在安装 Python 依赖...
    set MSG_PIP_DONE=依赖安装完成。
    set MSG_TOKENIZER_CHECK=正在检查分词器...
    set MSG_TOKENIZER_TRAIN=分词器未找到，正在训练...
    set MSG_TOKENIZER_DONE=分词器训练完成。
    set MSG_ENV_DONE=环境安装完成！正在启动...
    set MSG_PRESS_KEY=按任意键退出...
)

echo.
echo ========================================
echo %MSG_PYTHON_CHECK%
python --version >nul 2>&1
if errorlevel 1 (
    echo %MSG_PYTHON_FAIL%
    echo %MSG_PYTHON_DOWNLOAD%
    pause
    exit /b 1
)
echo [√] Python is installed / Python 已安装

:: 2. Create virtual environment / 创建虚拟环境
if not exist venv (
    echo [√] %MSG_VENV_CREATE%
    python -m venv venv
)
echo [√] %MSG_VENV_READY%

:: 3. Activate virtual environment / 激活虚拟环境
echo [√] %MSG_VENV_ACTIVATE%
call venv\Scripts\activate.bat

:: 4. Upgrade pip / 升级 pip
echo [√] %MSG_PIP_UPGRADE%
python -m pip install --upgrade pip -q

:: 5. Install dependencies / 安装依赖
echo [√] %MSG_PIP_INSTALL%
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo [√] %MSG_PIP_DONE%

:: 6. Check tokenizer / 检查分词器
echo [√] %MSG_TOKENIZER_CHECK%
if not exist tokenizer\our_bpe.model (
    echo [!] %MSG_TOKENIZER_TRAIN%
    python tokenizer_train.py
    echo [√] %MSG_TOKENIZER_DONE%
)

echo.
echo ========================================
echo    %MSG_ENV_DONE%
echo ========================================
echo.

:: 7. Launch / 启动
if exist start.ps1 (
    powershell -ExecutionPolicy Bypass -File start.ps1
) else if exist start.bat (
    call start.bat
) else (
    python chat_web.py
)

echo.
echo %MSG_PRESS_KEY%
pause