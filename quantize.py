# quantize.py —— 模型量化工具（带加载容错）
import torch
import sentencepiece as spm
import os
import glob
from model import MiniChat
from config import Config

cfg = Config()

# ---------- 分词器加载 ----------
sp = spm.SentencePieceProcessor()
tokenizer_path = f"tokenizer/{cfg.tokenizer_prefix}.model"
if not os.path.exists(tokenizer_path):
    print(f"❌ 分词器文件不存在: {tokenizer_path}")
    print("请先运行: python tokenizer_train.py")
    exit(1)
try:
    sp.load(tokenizer_path)
except Exception as e:
    print(f"❌ 分词器加载失败: {e}")
    exit(1)
vocab_size = sp.get_piece_size()

# ---------- 模型加载 ----------
model = MiniChat(vocab_size)
ckpts = glob.glob(f"{cfg.checkpoint_dir}/*.pt")
if not ckpts:
    print("❌ 未找到任何检查点，无法量化")
    exit(1)

latest = max(ckpts, key=os.path.getctime)
try:
    checkpoint = torch.load(latest, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ 已加载检查点: {latest}")
except Exception as e:
    print(f"❌ 检查点加载失败: {e}")
    exit(1)
model.eval()

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

output_path = "checkpoints/quantized_model.pt"
torch.save(quantized_model.state_dict(), output_path)
print(f"✅ 量化模型已保存到 {output_path}")