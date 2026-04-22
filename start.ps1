[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)

# 辅助函数：带颜色输出
function Write-ColorLine($Text, $Color) {
    Write-Host $Text -ForegroundColor $Color
}

# 检查 Python
function Check-Python {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        Write-ColorLine "✅ Python 已就绪" Green
    } else {
        Write-ColorLine "❌ 未找到 python，请先安装 Python 3.8+" Red
        Read-Host "按回车键退出"
        exit 1
    }
}

# 激活虚拟环境
function Activate-Venv {
    if (Test-Path "venv\Scripts\Activate.ps1") {
        . .\venv\Scripts\Activate.ps1
        Write-ColorLine "✅ 已激活本地虚拟环境" Green
    } elseif (Test-Path "..\venv\Scripts\Activate.ps1") {
        . ..\venv\Scripts\Activate.ps1
        Write-ColorLine "✅ 已激活上层虚拟环境" Green
    } else {
        Write-ColorLine "⚠️  未找到虚拟环境，使用系统 Python" Yellow
    }
}

# 检查 PyTorch
function Check-PyTorch {
    python -c "import torch" 2>$null
    if ($?) {
        $ver = python -c "import torch; print(torch.__version__)"
        Write-ColorLine "✅ PyTorch 已安装 (版本 $ver)" Green
    } else {
        Write-ColorLine "❌ PyTorch 未安装，请运行: pip install -r requirements.txt" Red
        Read-Host "按回车键退出"
        exit 1
    }
}

# 检查分词器
function Check-Tokenizer {
    if (Test-Path "tokenizer\our_bpe.model") {
        Write-ColorLine "✅ 分词器已存在" Green
    } else {
        Write-ColorLine "⚠️  分词器缺失，开始自动训练..." Yellow
        python tokenizer_train.py
        Write-ColorLine "✅ 分词器训练完成" Green
    }
}

# 检查模型
function Check-Model {
    $ckpt = (Get-ChildItem "checkpoints\*.pt" -ErrorAction SilentlyContinue).Count
    $lora = (Get-ChildItem "lora_weights\*.pt" -ErrorAction SilentlyContinue).Count

    if ($ckpt -gt 0 -or $lora -gt 0) {
        Write-ColorLine "✅ 已找到模型检查点" Green
        return
    }

    Write-ColorLine "⚠️  未找到任何模型检查点" Yellow
    $answer = Read-Host "是否立即开始训练模型？(y/n)"
    if ($answer -match '^[Yy]') {
        python train.py
    } else {
        Write-ColorLine "⚠️  将使用随机初始化模型（效果会很差）" Yellow
    }
}

# 独立工具菜单
function Show-ToolsMenu {
    do {
        Write-Host ""
        Write-ColorLine "🧰 独立工具菜单" Blue
        Write-Host "  1) 联网搜索"
        Write-Host "  2) 对话历史导出"
        Write-Host "  3) 模型量化"
        Write-Host "  4) 数据集预览"
        Write-Host "  5) 长期记忆测试"
        Write-Host "  6) 返回主菜单"
        $choice = Read-Host "请输入数字"

        switch ($choice) {
            '1' { python web_search.py }
            '2' { python export_history.py }
            '3' { python quantize.py }
            '4' { python preview_data.py }
            '5' { python recall.py }
            '6' { return }
            default { Write-ColorLine "无效输入" Red }
        }
        if ($choice -ne '6') { Read-Host "按回车键继续..." }
    } while ($choice -ne '6')
}

# ============================================
# 主流程
# ============================================
Write-ColorLine "========================================" Blue
Write-ColorLine "   🤖 我的专属 AI 助手 启动器 (Windows)" Green
Write-ColorLine "========================================" Blue

Check-Python
Activate-Venv
Check-PyTorch
Check-Tokenizer
Check-Model

do {
    Write-Host ""
    Write-ColorLine "请选择启动模式:" Blue
    Write-Host "  1) Web 界面 (Gradio) - 推荐"
    Write-Host "  2) 命令行对话 (CLI)"
    Write-Host "  3) API 后端服务 (FastAPI)"
    Write-Host "  4) 测试 RAG 模块"
    Write-Host "  5) 人设对话模式"
    Write-Host "  6) 🧰 独立工具"
    Write-Host "  7) 退出"
    $mainChoice = Read-Host "请输入数字 (1-7)"

    switch ($mainChoice) {
        '1' {
            Write-ColorLine "🚀 正在启动 Web 界面..." Green
            python chat_web.py
            break
        }
        '2' {
            Write-ColorLine "💬 正在启动命令行对话..." Green
            python chat_cli.py
            break
        }
        '3' {
            Write-ColorLine "🌐 正在启动 API 后端 (端口 8000)..." Green
            if (Test-Path "backend\main.py") {
                Set-Location backend
                python main.py
                Set-Location ..
            } else {
                Write-ColorLine "❌ backend 目录或 main.py 不存在" Red
            }
            break
        }
        '4' {
            Write-ColorLine "📚 测试 RAG 模块..." Green
            python -c "from rag_module import RAGModule; print('✅ RAG 模块正常')"
            break
        }
        '5' {
            Write-ColorLine "🎭 启动人设对话模式..." Green
            python persona_chat.py
            break
        }
        '6' {
            Show-ToolsMenu
        }
        '7' {
            Write-ColorLine "👋 再见！" Yellow
            exit 0
        }
        default {
            Write-ColorLine "无效输入，请重新选择" Red
        }
    }
} while ($true)