# persona_chat.py —— 独立人设对话工具，不修改任何现有文件
import torch
import sentencepiece as spm
from config import Config
from model import MiniChat
from utils import format_chat_prompt
import os
import glob

# 读取人设文件
persona_file = "persona.txt"
if os.path.exists(persona_file):
    with open(persona_file, "r", encoding="utf-8") as f:
        PERSONA = f.read().strip()
    print(f"🎭 已加载人设: {PERSONA[:50]}...")
else:
    PERSONA = "你是一个有帮助的助手。"
    print("⚠️ 未找到 persona.txt，使用默认人设。")

# 自定义提示模板（融入人设）
def persona_format_prompt(instruction):
    return f"{PERSONA}\n### 用户: {instruction}\n### 助手: "

# 加载模型和分词器（与 chat_cli.py 完全相同的逻辑）
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

print("\n🎭 人设对话模式（输入 quit 退出）")
print(f"当前人设: {PERSONA}\n")

history = ""
while True:
    user_input = input("你: ")
    if user_input.lower() == 'quit':
        break
    prompt = persona_format_prompt(user_input)
    full_prompt = history + prompt
    input_ids = torch.tensor([sp.bos_id()] + sp.encode(full_prompt, out_type=int), device=device).unsqueeze(0)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200)
    response_ids = output_ids[0, input_ids.shape[1]:].tolist()
    response = sp.decode(response_ids)
    # 清理可能残留的提示格式
    if "### 助手:" in response:
        response = response.split("### 助手:")[-1].strip()
    print(f"AI: {response}")
    history += prompt + response + "\n"
    if len(history) > 1000:
        history = history[-1000:]