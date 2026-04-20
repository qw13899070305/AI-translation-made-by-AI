import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from config import Config
from dataset import get_dataloader
from model import MiniChat
from lora import apply_lora_to_model, mark_only_lora_as_trainable
from utils import ensure_dir
import sentencepiece as spm
from tqdm import tqdm

cfg = Config()
ensure_dir(cfg.checkpoint_dir)
ensure_dir(cfg.lora_checkpoint_dir)

sp = spm.SentencePieceProcessor()
sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
vocab_size = sp.get_piece_size()

model = MiniChat(vocab_size).to(cfg.device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

if cfg.use_lora:
    apply_lora_to_model(model)
    mark_only_lora_as_trainable(model)
    print("已应用 LoRA，仅训练 LoRA 参数。")

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)
train_loader = get_dataloader()

for epoch in range(1, cfg.epochs+1):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        logits, loss, _ = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
    if epoch % cfg.save_every == 0:
        if cfg.use_lora:
            save_path = f"{cfg.lora_checkpoint_dir}/lora_epoch_{epoch}.pt"
            lora_state = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
            torch.save(lora_state, save_path)
        else:
            save_path = f"{cfg.checkpoint_dir}/epoch_{epoch}.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, save_path)
        print(f"已保存检查点到 {save_path}")