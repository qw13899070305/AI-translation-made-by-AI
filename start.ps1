# ============================================
# 我的专属 AI 助手 - Windows PowerShell 启动脚本（完整版）
# ============================================

# 设置控制台编码为 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

# 颜色定义函数
function Write-ColorLine($text, $color) {
    Write-Host $text -ForegroundColor $color
}

Write-ColorLine "========================================" Blue
Write-ColorLine "   🤖 我的专属 AI 助手 启动器 (Windows)" Green
Write-ColorLine "========================================" Blue

# 检查 Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-ColorLine "[错误] 未找到 python，请先安装 Python 3.8+" Red
    Read-Host "按回车键退出"
    exit 1
}

# 激活虚拟环境（如果存在）
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-ColorLine "[√] 发现虚拟环境，正在激活..." Green
    . .\venv\Scripts\Activate.ps1
} elseif (Test-Path "..\venv\Scripts\Activate.ps1") {
    Write-ColorLine "[√] 发现虚拟环境，正在激活..." Green
    . ..\venv\Scripts\Activate.ps1
} else {
    Write-ColorLine "[!] 未找到虚拟环境，使用系统 Python" Yellow
}

# 检查 PyTorch
Write-ColorLine "🔍 检查依赖..." Blue
$torchCheck = python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')" 2>$null
if (-not $?) {
    Write-ColorLine "[错误] PyTorch 未安装，请先运行: pip install -r requirements.txt" Red
    Read-Host "按回车键退出"
    exit 1
} else {
    Write-Host $torchCheck
}

# 检查分词器
if (-not (Test-Path "tokenizer\our_bpe.model")) {
    Write-ColorLine "[!] 分词器未找到，正在自动训练..." Yellow
    python tokenizer_train.py
    if (-not $?) {
        Write-ColorLine "[错误] 分词器训练失败" Red
        Read-Host "按回车键退出"
        exit 1
    }
}

# 检查模型检查点
$ckptCount = (Get-ChildItem "checkpoints\*.pt" -ErrorAction SilentlyContinue).Count
$loraCount = (Get-ChildItem "lora_weights\*.pt" -ErrorAction SilentlyContinue).Count
if ($ckptCount -eq 0 -and $loraCount -eq 0) {
    Write-ColorLine "[!] 未找到任何模型检查点，您需要先训练模型。" Yellow
    $answer = Read-Host "是否立即开始训练？(y/n)"
    if ($answer -eq 'y' -or $answer -eq 'Y') {
        python train.py
        if (-not $?) {
            Write-ColorLine "[错误] 训练失败" Red
            Read-Host "按回车键退出"
            exit 1
        }
    } else {
        Write-ColorLine "[!] 将使用随机初始化模型（效果可能不佳）。" Yellow
    }
}

# 显示菜单
Write-Host ""
Write-ColorLine "请选择启动模式:" Blue
Write-Host "  1) Web 界面 (Gradio) - 推荐，支持图文和 RAG"
Write-Host "  2) 命令行对话 (CLI)"
Write-Host "  3) API 后端服务 (FastAPI) - 供手机端调用"
Write-Host "  4) 仅测试 RAG 模块"
Write-Host "  5) 退出"
$choice = Read-Host "输入数字 (1-5)"

switch ($choice) {
    '1' {
        Write-ColorLine "🚀 启动 Web 界面..." Green
        python chat_web.py
    }
    '2' {
        Write-ColorLine "💬 启动命令行对话..." Green
        python chat_cli.py
    }
    '3' {
        Write-ColorLine "🌐 启动 API 后端服务 (端口 8000)..." Green
        if (Test-Path "backend\main.py") {
            Set-Location backend
            python main.py
            Set-Location ..
        } else {
            Write-ColorLine "[错误] backend 目录不存在" Red
            Read-Host "按回车键退出"
            exit 1
        }
    }
    '4' {
        Write-ColorLine "📚 测试 RAG 模块..." Green
        python -c "from rag_module import RAGModule; r = RAGModule(); print('RAG 模块加载成功！')"
    }
    '5' {
        Write-ColorLine "👋 再见！" Yellow
        exit 0
    }
    default {
        Write-ColorLine "无效选择，退出。" Red
        Read-Host "按回车键退出"
        exit 1
    }
}

Read-Host "按回车键退出"