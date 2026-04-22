import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sentencepiece as spm
import uvicorn
import glob
from config import Config
from model import MiniChat
from utils import format_chat_prompt

cfg = Config()
device = cfg.device

# ---------- 分词器加载（带错误处理） ----------
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

# ---------- 模型加载（带容错） ----------
model = MiniChat(vocab_size).to(device)
ckpts = glob.glob(f"{cfg.checkpoint_dir}/*.pt")
if ckpts:
    latest = max(ckpts, key=os.path.getctime)
    try:
        checkpoint = torch.load(latest, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 已加载检查点: {latest}")
    except Exception as e:
        print(f"⚠️ 检查点加载失败: {e}，使用随机初始化模型")
else:
    print("⚠️ 未找到检查点，使用随机初始化模型")
model.eval()
print(f"✅ 模型加载完成，设备：{device}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        prompt = format_chat_prompt(request.message)
        input_ids = torch.tensor(
            [sp.bos_id()] + sp.encode(prompt, out_type=int),
            device=device
        ).unsqueeze(0)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=200)
        response_ids = output_ids[0, input_ids.shape[1]:].tolist()
        response = sp.decode(response_ids)
        if "### 助手:" in response:
            response = response.split("### 助手:")[-1].strip()
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)