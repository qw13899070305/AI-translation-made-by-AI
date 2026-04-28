import os, torch
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
model = MiniChat(sp.get_piece_size()).to(cfg.device)

if cfg.use_lora:
    apply_lora_to_model(model)
    mark_only_lora_as_trainable(model)

# 优化器选择
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

best_loss, patience, no_improve = float('inf'), 2, 0
for epoch in range(1, cfg.epochs+1):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        with autocast(enabled=(cfg.use_amp and cfg.device == "cuda")):
            _, loss, _, _ = model(inputs, targets)
        if torch.isnan(loss): continue
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss, no_improve = avg_loss, 0
        if epoch % cfg.save_every == 0:
            save_dict = {'epoch': epoch, 'loss': avg_loss}
            if cfg.use_lora:
                save_dict['model_state_dict'] = model.state_dict()
                save_dict['lora_config'] = {'r': cfg.lora_r, 'alpha': cfg.lora_alpha}
                save_path = f"{cfg.lora_checkpoint_dir}/lora_epoch_{epoch}.pt"
            else:
                save_dict['model_state_dict'] = model.state_dict()
                save_path = f"{cfg.checkpoint_dir}/epoch_{epoch}.pt"
            torch.save(save_dict, save_path)
            print(f"Checkpoint saved to {save_path}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break