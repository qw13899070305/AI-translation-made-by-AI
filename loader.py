# loader.py —— 统一模型与分词器加载
import os
import glob
import torch
import sentencepiece as spm
from config import Config
from model import MiniChat

def load_model_and_tokenizer(lang="en"):
    cfg = Config()
    device = cfg.device

    # 分词器
    sp = spm.SentencePieceProcessor()
    tokenizer_path = f"tokenizer/{cfg.tokenizer_prefix}.model"
    if not os.path.exists(tokenizer_path):
        msg = f"❌ Tokenizer not found: {tokenizer_path}\nPlease run: python tokenizer_train.py"
        raise FileNotFoundError(msg)
    sp.load(tokenizer_path)

    # 模型
    model = MiniChat(sp.get_piece_size()).to(device)
    ckpts = glob.glob(f"{cfg.checkpoint_dir}/*.pt")
    if ckpts:
        latest = max(ckpts, key=os.path.getctime)
        checkpoint = torch.load(latest, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint: {latest}")
    else:
        print("⚠️  No checkpoint found, using random weights.")

    model.eval()
    return model, sp, device