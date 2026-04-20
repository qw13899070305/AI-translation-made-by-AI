import torch
import sentencepiece as spm
from config import Config
from model import MiniChat
from utils import format_chat_prompt
import os
import glob

cfg = Config()
device = cfg.device

sp = spm.SentencePieceProcessor()
sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
vocab_size = sp.get_piece_size()

model = MiniChat(vocab_size).to(device)
ckpts = glob.glob(f"{cfg.checkpoint_dir}/*.pt")
if ckpts:
    latest = max(ckpts, key=os.path.getctime)
    checkpoint = torch.load(latest, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载检查点: {latest}")
else:
    print("警告：未找到检查点，使用随机初始化模型。")
model.eval()

print("命令行对话模式（输入 quit 退出）")
history = ""
while True:
    user_input = input("\n你: ")
    if user_input.lower() == 'quit':
        break
    prompt = format_chat_prompt(user_input)
    full_prompt = history + prompt
    input_ids = torch.tensor([sp.bos_id()] + sp.encode(full_prompt, out_type=int), device=device).unsqueeze(0)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200)
    response_ids = output_ids[0, input_ids.shape[1]:].tolist()
    response = sp.decode(response_ids)
    if "### 助手:" in response:
        response = response.split("### 助手:")[-1].strip()
    print(f"AI: {response}")
    history += prompt + response + "\n"
    if len(history) > 1000:
        history = history[-1000:]