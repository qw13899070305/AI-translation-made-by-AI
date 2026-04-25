[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)

function Write-ColorLine($Text, $Color) { Write-Host $Text -ForegroundColor $Color }

function Check-Python {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        Write-ColorLine "✅ Python 已就绪" Green
    } else {
        Write-ColorLine "❌ 未找到 python，请先安装 Python 3.8+" Red
        Read-Host "按回车键退出"
        exit 1
    }
}

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

function Check-Tokenizer {
    if (Test-Path "tokenizer\our_bpe.model") {
        Write-ColorLine "✅ 分词器已存在" Green
    } else {
        Write-ColorLine "⚠️  分词器缺失，开始自动训练..." Yellow
        python tokenizer_train.py
        Write-ColorLine "✅ 分词器训练完成" Green
    }
}

function Check-Model {
    $ckpt = (Get-ChildItem "checkpoints\*.pt" -ErrorAction SilentlyContinue).Count
    $lora = (Get-ChildItem "lora_weights\*.pt" -ErrorAction SilentlyContinue).Count
    if ($ckpt -gt 0 -or $lora -gt 0) { return $true }
    return $false
}

function Show-ToolsMenu {
    do {
        Write-Host ""
        Write-ColorLine "🧰 独立工具菜单" Blue
        Write-Host "  1) 联网搜索"
        Write-Host "  2) 对话历史导出"
        Write-Host "  3) 模型量化"
        Write-Host "  4) 数据集预览"
        Write-Host "  5) 长期记忆测试"
        Write-Host "  6) 数据扩充（下载多源数据）"
        Write-Host "  7) 多任务学习训练"
        Write-Host "  8) 持续学习训练"
        Write-Host "  9) 返回主菜单"
        $choice = Read-Host "请输入数字 (1-9)"
        switch ($choice) {
            '1' { python web_search.py }
            '2' { python export_history.py }
            '3' { python quantize.py }
            '4' { python preview_data.py }
            '5' { python recall.py }
            '6' { python enhanced_data_loader.py }
            '7' { python multitask_trainer.py }
            '8' { python continual_trainer.py }
            '9' { return }
            default { Write-ColorLine "无效输入" Red }
        }
        if ($choice -ne '9') { Read-Host "按回车键继续..." }
    } while ($choice -ne '9')
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

if (-not (Check-Model)) {
    Write-ColorLine "⚠️  未找到任何模型检查点" Yellow
    Write-ColorLine "💡 建议：选择 0 进入配置中心调整参数，然后选 6 训练模型" Yellow
}

do {
    Write-Host ""
    Write-ColorLine "请选择启动模式:" Blue
    Write-Host "  0) 🛠️  配置管理中心（建议训练前先配置）"
    Write-Host "  1) 🚀 启动 Web 界面"
    Write-Host "  2) 💬 启动命令行对话"
    Write-Host "  3) 🌐 启动 API 后端服务"
    Write-Host "  4) 📚 测试 RAG 模块"
    Write-Host "  5) 🎭 启动人设对话模式"
    Write-Host "  6) 📦 训练模型"
    Write-Host "  7) 🔧 训练分词器"
    Write-Host "  8) 🧰 独立工具"
    Write-Host "  9) 👋 退出"
    $mainChoice = Read-Host "请输入数字 (0-9)"

    switch ($mainChoice) {
        '0' {
            Write-ColorLine "🛠️  启动配置管理中心..." Green
            python config_manager.py
        }
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
            if (Check-Model) {
                Write-ColorLine "⚠️  已有模型检查点，继续训练将覆盖旧模型。确认？(y/n)" Yellow
                $confirm = Read-Host
                if ($confirm -match '^[Yy]') {
                    python train.py
                }
            } else {
                Write-ColorLine "📦 开始训练模型..." Green
                python train.py
            }
        }
        '7' {
            Write-ColorLine "🔧 训练分词器..." Green
            python tokenizer_train.py
        }
        '8' {
            Show-ToolsMenu
        }
        '9' {
            Write-ColorLine "👋 再见！" Yellow
            exit 0
        }
        default {
            Write-ColorLine "无效输入，请重新选择" Red
        }
    }
} while ($true)