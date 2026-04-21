# quantize.py —— 模型量化工具
import torch
from model import MiniChat
import sentencepiece as spm
from config import Config

cfg = Config()
sp = spm.SentencePieceProcessor()
sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
vocab_size = sp.get_piece_size()

model = MiniChat(vocab_size)
checkpoint = torch.load("checkpoints/epoch_5.pt", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 动态量化（最简单，无需校准数据）
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 保存量化模型
torch.save(quantized_model.state_dict(), "checkpoints/quantized_model.pt")
print("✅ 量化模型已保存")