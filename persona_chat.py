import torch
import sentencepiece as spm
import os
import glob
from config import Config
from model import MiniChat

# 读取人设文件
persona_file = "persona.txt"
if os.path.exists(persona_file):
    with open(persona_file, "r", encoding="utf-8") as f:
        PERSONA = f.read().strip()
    print(f"🎭 已加载人设: {PERSONA[:50]}...")
else:
    PERSONA = "你是一个有帮助的助手。"
    print("⚠️ 未找到 persona.txt，使用默认人设。")

def persona_format_prompt(instruction):
    return f"{PERSONA}\n### 用户: {instruction}\n### 助手: "

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
    if "### 助手:" in response:
        response = response.split("### 助手:")[-1].strip()
    print(f"AI: {response}")
    history += prompt + response + "\n"
    if len(history) > 1000:
        history = history[-1000:]