import torch
import torch.nn as nn
import math
from config import Config

cfg = Config()

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.linear(x)
        if self.r > 0:
            lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            result = result + lora_out
        return result

def apply_lora_to_model(model, target_modules=None):
    if target_modules is None:
        target_modules = cfg.lora_target_modules
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            parent = model
            name_parts = name.split('.')
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            original_linear = getattr(parent, name_parts[-1])
            lora_layer = LoRALinear(original_linear.in_features, original_linear.out_features, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
            lora_layer.linear.weight.data.copy_(original_linear.weight.data)
            if original_linear.bias is not None:
                lora_layer.linear.bias.data.copy_(original_linear.bias.data)
            setattr(parent, name_parts[-1], lora_layer)

def mark_only_lora_as_trainable(model):
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False