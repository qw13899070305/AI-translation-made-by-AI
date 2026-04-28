import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from config import Config

cfg = Config()

class QFormer(nn.Module):
    def __init__(self, vision_embed_dim, proj_dim, num_queries=32, num_layers=2, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, proj_dim))
        self.vision_proj = nn.Linear(vision_embed_dim, proj_dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.ffn = nn.Sequential(nn.Linear(proj_dim, proj_dim * 4), nn.GELU(), nn.Linear(proj_dim * 4, proj_dim))

    def forward(self, vision_features):
        vision_features = self.vision_proj(vision_features)
        B = vision_features.shape[0]
        query_tokens = self.query_tokens.expand(B, -1, -1)
        attn_out, _ = self.cross_attention(query_tokens, vision_features, vision_features)
        query_tokens = self.norm1(query_tokens + attn_out)
        ffn_out = self.ffn(query_tokens)
        query_tokens = self.norm2(query_tokens + ffn_out)
        return query_tokens

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(cfg.vision_encoder_name)
        self.processor = CLIPImageProcessor.from_pretrained(cfg.vision_encoder_name)
        self.qformer = QFormer(cfg.vision_embed_dim, cfg.proj_dim, cfg.num_queries, cfg.qformer_layers, cfg.qformer_heads)
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(cfg.device)
        outputs = self.vision_model(**inputs)
        vision_features = outputs.last_hidden_state[:, 1:, :]
        return self.qformer(vision_features)