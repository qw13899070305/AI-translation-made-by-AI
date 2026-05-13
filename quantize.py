import torch
import sentencepiece as spm
import os
import glob
from model import MiniChat
from config import Config

cfg = Config()
sp = spm.SentencePieceProcessor()
tokenizer_path = f"tokenizer/{cfg.tokenizer_prefix}.model"
if not os.path.exists(tokenizer_path):
    print(f"❌ Tokenizer not found: {tokenizer_path}")
    exit(1)
try:
    sp.load(tokenizer_path)
except Exception as e:
    print(f"❌ Tokenizer load failed: {e}")
    exit(1)
vocab_size = sp.get_piece_size()

model = MiniChat(vocab_size)
ckpts = glob.glob(f"{cfg.checkpoint_dir}/*.pt")
if not ckpts:
    print("❌ No checkpoints found")
    exit(1)
latest = max(ckpts, key=os.path.getctime)
try:
    checkpoint = torch.load(latest, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded: {latest}")
except Exception as e:
    print(f"❌ Load failed: {e}")
    exit(1)

model.eval()

# 使用新版动态量化 API
try:
    from torch.ao.quantization import quantize_dynamic
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
except ImportError:
    # 回退到旧版
    import torch.quantization
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

output_path = "checkpoints/quantized_model.pt"
torch.save(quantized_model.state_dict(), output_path)
print(f"✅ Quantized model saved to {output_path}")