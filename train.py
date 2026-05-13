import os, torch
import torch.nn.functional as F          # 添加这一行
from torch.cuda.amp import GradScaler, autocast
from config import Config
from dataset import get_dataloader
from model import MiniChat
from lora import apply_lora_to_model, mark_only_lora_as_trainable
from utils import ensure_dir
from muon import Muon, MuonClip
import sentencepiece as spm
from tqdm import tqdm

cfg = Config()
ensure_dir(cfg.checkpoint_dir)
ensure_dir(cfg.lora_checkpoint_dir)

sp = spm.SentencePieceProcessor()
sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
vocab_size = sp.get_piece_size()
model = MiniChat(vocab_size).to(cfg.device)

if cfg.use_lora:
    apply_lora_to_model(model)
    mark_only_lora_as_trainable(model)

# 优化器
if cfg.use_muon:
    muon_params, adam_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim >= 2: muon_params.append(p)
        else: adam_params.append(p)
    if cfg.use_muon_clip:
        optimizer = MuonClip([
            {'params': muon_params, 'muon_wd': 0.0, 'adam_wd': 0.01},
            {'params': adam_params, 'muon_wd': 0.0, 'adam_wd': 0.01}
        ], lr=cfg.learning_rate, clip_grad=cfg.muon_clip_grad, clip_update=cfg.muon_clip_update)
    else:
        optimizer = Muon([
            {'params': muon_params, 'muon_wd': 0.0, 'adam_wd': 0.01},
            {'params': adam_params, 'muon_wd': 0.0, 'adam_wd': 0.01}
        ], lr=cfg.learning_rate)
else:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)

scaler = GradScaler(enabled=(cfg.use_amp and cfg.device == "cuda"))

train_loader = get_dataloader()

# HTA 调度器
scheduler = None
if cfg.use_hta:
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    total_steps = cfg.epochs * len(train_loader)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.hta_warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - cfg.hta_warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.hta_warmup_steps])

best_loss, patience, no_improve = float('inf'), 2, 0
start_epoch = 0
resume_path = None  # 可设置断点续训路径

if resume_path and os.path.exists(resume_path):
    checkpoint = torch.load(resume_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float('inf'))
    print(f"Resumed from {resume_path}, epoch {start_epoch}")

for epoch in range(start_epoch, cfg.epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
    for step, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        with autocast(dtype=torch.float8_e4m3fn if cfg.use_fp8 else torch.float16,
                      enabled=(cfg.use_amp and cfg.device == "cuda")):
            _, loss, _, mtp_preds = model(inputs, targets=targets)

        # ========== 添加 MTP 辅助损失 ==========
        if mtp_preds is not None and cfg.use_mtp:
            mtp_loss = 0.0
            for i, pred in enumerate(mtp_preds):
                shift = i + 1
                if targets.shape[1] > shift:
                    mtp_loss += F.cross_entropy(
                        pred[:, :-shift, :].reshape(-1, vocab_size),
                        targets[:, shift:].reshape(-1)
                    )
            if mtp_loss != 0.0:
                mtp_loss = mtp_loss / len(mtp_preds)
                loss = loss + 0.3 * mtp_loss
        # =====================================

        if torch.isnan(loss):
            continue
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 更新 MoE 动态偏置
        if cfg.use_moe:
            for block in model.blocks:
                if hasattr(block.ffn, 'update_bias'):
                    block.ffn.update_bias()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 保存检查点（包含优化器状态）
    if (epoch+1) % cfg.save_every == 0:
        save_dict = {
            'epoch': epoch+1,
            'loss': avg_loss,
            'best_loss': best_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()
        if cfg.use_lora:
            save_path = f"{cfg.lora_checkpoint_dir}/lora_epoch_{epoch+1}.pt"
        else:
            save_path = f"{cfg.checkpoint_dir}/epoch_{epoch+1}.pt"
        torch.save(save_dict, save_path)
        print(f"Checkpoint saved to {save_path}")