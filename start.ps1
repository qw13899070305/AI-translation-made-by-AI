[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)
function Write-ColorLine($Text, $Color) { Write-Host $Text -ForegroundColor $Color }

# ---------- 语言选择 ----------
Write-Host "========================================"
Write-Host "   🤖 My Own AI Assistant / 我的专属 AI 助手"
Write-Host "========================================"
Write-Host "  Please select language / 请选择语言:"
Write-Host "  1) English"
Write-Host "  2) 中文"
$langChoice = Read-Host "Enter choice (1 or 2)"

if ($langChoice -eq "1") {
    $MSG_PYTHON_FAIL = "❌ Python not found. Please install Python 3.8+"
    $MSG_VENV_ACTIVATE = "✅ Virtual environment activated."
    $MSG_VENV_NOT_FOUND = "⚠️  Virtual environment not found, using system Python."
    $MSG_PYTORCH_FAIL = "❌ PyTorch not installed. Please run: pip install -r requirements.txt"
    $MSG_TOKENIZER_TRAIN = "⚠️  Tokenizer not found, training..."
    $MSG_MAIN_MENU = "Please select a startup mode:"
    $MSG_MENU_CONFIG = "🛠️  Configuration Center"
    $MSG_MENU_WEB = "🚀 Launch Web Interface (text/image/voice/doc)"
    $MSG_MENU_CLI = "💬 Launch Command Line Chat"
    $MSG_MENU_API = "🌐 Launch API Backend (port 8000)"
    $MSG_MENU_RAG = "📚 Test RAG Knowledge Base"
    $MSG_MENU_PERSONA = "🎭 Launch Persona Chat"
    $MSG_MENU_TRAIN = "📦 Train Model (supports Muon/AMP)"
    $MSG_MENU_TOKENIZER = "🔧 Train Tokenizer"
    $MSG_MENU_TOOLS = "🧰 Standalone Tools"
    $MSG_MENU_EXIT = "👋 Exit"
    $MSG_TOOL1 = "Web Search"; $MSG_TOOL2 = "Export History"; $MSG_TOOL3 = "Model Quantization"
    $MSG_TOOL4 = "Dataset Preview"; $MSG_TOOL5 = "Memory Test"; $MSG_TOOL6 = "Data Expansion"
    $MSG_TOOL7 = "Multi-task Learning"; $MSG_TOOL8 = "Continual Learning"; $MSG_TOOL9 = "🎓 MOPD Distillation"
    $MSG_TOOL0 = "Return to Main Menu"; $MSG_GOODBYE = "👋 Goodbye!"
} else {
    $MSG_PYTHON_FAIL = "❌ 未找到 Python，请先安装 Python 3.8+"
    $MSG_VENV_ACTIVATE = "✅ 已激活虚拟环境。"
    $MSG_VENV_NOT_FOUND = "⚠️  未找到虚拟环境，使用系统 Python。"
    $MSG_PYTORCH_FAIL = "❌ PyTorch 未安装，请运行: pip install -r requirements.txt"
    $MSG_TOKENIZER_TRAIN = "⚠️  分词器未找到，正在训练..."
    $MSG_MAIN_MENU = "请选择启动模式:"
    $MSG_MENU_CONFIG = "🛠️  配置管理中心"
    $MSG_MENU_WEB = "🚀 启动 Web 界面（支持图文/语音/文档上传）"
    $MSG_MENU_CLI = "💬 启动命令行对话"
    $MSG_MENU_API = "🌐 启动 API 后端服务 (端口 8000)"
    $MSG_MENU_RAG = "📚 测试 RAG 知识库模块"
    $MSG_MENU_PERSONA = "🎭 启动人设对话模式"
    $MSG_MENU_TRAIN = "📦 训练模型（支持 Muon/AMP）"
    $MSG_MENU_TOKENIZER = "🔧 训练分词器"
    $MSG_MENU_TOOLS = "🧰 独立工具"
    $MSG_MENU_EXIT = "👋 退出"
    $MSG_TOOL1 = "联网搜索"; $MSG_TOOL2 = "对话导出"; $MSG_TOOL3 = "模型量化"
    $MSG_TOOL4 = "数据集预览"; $MSG_TOOL5 = "长期记忆测试"; $MSG_TOOL6 = "数据扩充"
    $MSG_TOOL7 = "多任务学习"; $MSG_TOOL8 = "持续学习"; $MSG_TOOL9 = "🎓 MOPD 多教师蒸馏"
    $MSG_TOOL0 = "返回主菜单"; $MSG_GOODBYE = "👋 再见！"
}

function Check-Python { if (Get-Command python -ErrorAction SilentlyContinue) { Write-ColorLine $MSG_VENV_ACTIVATE Green } else { Write-ColorLine $MSG_PYTHON_FAIL Red; Read-Host; exit 1 } }
function Activate-Venv { if (Test-Path "venv\Scripts\Activate.ps1") { . .\venv\Scripts\Activate.ps1; Write-ColorLine $MSG_VENV_ACTIVATE Green } elseif (Test-Path "..\venv\Scripts\Activate.ps1") { . ..\venv\Scripts\Activate.ps1; Write-ColorLine $MSG_VENV_ACTIVATE Green } else { Write-ColorLine $MSG_VENV_NOT_FOUND Yellow } }
function Check-PyTorch { python -c "import torch" 2>$null; if ($?) { $ver = python -c "import torch; print(torch.__version__)"; Write-ColorLine "✅ PyTorch $ver" Green } else { Write-ColorLine $MSG_PYTORCH_FAIL Red; Read-Host; exit 1 } }
function Check-Tokenizer { if (Test-Path "tokenizer\our_bpe.model") { Write-ColorLine "✅ Tokenizer exists." Green } else { Write-ColorLine $MSG_TOKENIZER_TRAIN Yellow; python tokenizer_train.py; Write-ColorLine "✅ Tokenizer training completed." Green } }
function Check-Model { $ckpt = (Get-ChildItem "checkpoints\*.pt" -ErrorAction SilentlyContinue).Count; $lora = (Get-ChildItem "lora_weights\*.pt" -ErrorAction SilentlyContinue).Count; if ($ckpt -gt 0 -or $lora -gt 0) { return $true } return $false }

function Show-ToolsMenu {
    do {
        Write-Host ""; Write-ColorLine "🧰 Tools Menu" Blue
        Write-Host "  1) $MSG_TOOL1"; Write-Host "  2) $MSG_TOOL2"; Write-Host "  3) $MSG_TOOL3"
        Write-Host "  4) $MSG_TOOL4"; Write-Host "  5) $MSG_TOOL5"; Write-Host "  6) $MSG_TOOL6"
        Write-Host "  7) $MSG_TOOL7"; Write-Host "  8) $MSG_TOOL8"; Write-Host "  9) $MSG_TOOL9"
        Write-Host "  0) $MSG_TOOL0"
        $choice = Read-Host ">"
        switch ($choice) {
            '1' { python web_search.py } '2' { python export_history.py } '3' { python quantize.py }
            '4' { python preview_data.py } '5' { python recall.py } '6' { python enhanced_data_loader.py }
            '7' { python multitask_trainer.py } '8' { python continual_trainer.py } '9' { python distill_mopd.py }
            '0' { return } default { Write-ColorLine "Invalid choice." Red }
        }
        if ($choice -ne '0') { Read-Host }
    } while ($choice -ne '0')
}

Write-ColorLine "========================================" Blue
Write-ColorLine "   🤖 My Own AI Assistant / 我的专属 AI 助手" Green
Write-ColorLine "========================================" Blue
Check-Python; Activate-Venv; Check-PyTorch; Check-Tokenizer
if (-not (Check-Model)) { Write-ColorLine "⚠️  No checkpoints found." Yellow; Write-ColorLine "💡 Choose 0 to enter Configuration Center, then 6 to train." Yellow }

do {
    Write-Host ""; Write-ColorLine $MSG_MAIN_MENU Blue
    Write-Host "  0) $MSG_MENU_CONFIG"; Write-Host "  1) $MSG_MENU_WEB"; Write-Host "  2) $MSG_MENU_CLI"
    Write-Host "  3) $MSG_MENU_API"; Write-Host "  4) $MSG_MENU_RAG"; Write-Host "  5) $MSG_MENU_PERSONA"
    Write-Host "  6) $MSG_MENU_TRAIN"; Write-Host "  7) $MSG_MENU_TOKENIZER"; Write-Host "  8) $MSG_MENU_TOOLS"
    Write-Host "  9) $MSG_MENU_EXIT"
    $mainChoice = Read-Host ">"
    switch ($mainChoice) {
        '0' { python config_manager.py }
        '1' { python chat_web.py; break }
        '2' { python chat_cli.py; break }
        '3' { if (Test-Path "backend\main.py") { Set-Location backend; python main.py; Set-Location .. } else { Write-ColorLine "❌ backend/main.py not found" Red }; break }
        '4' { python -c "from rag_module import RAGModule; print('✅ RAG OK')"; break }
        '5' { python persona_chat.py; break }
        '6' { if (Check-Model) { Write-ColorLine $MSG_CONFIRM Yellow; $confirm = Read-Host; if ($confirm -match '^[Yy]') { python train.py } } else { python train.py } }
        '7' { python tokenizer_train.py }
        '8' { Show-ToolsMenu }
        '9' { Write-ColorLine $MSG_GOODBYE Yellow; exit 0 }
        default { Write-ColorLine "Invalid choice." Red }
    }
} while ($true)