import torch
import os
from loader import load_model_and_tokenizer
from utils import format_chat_prompt

lang = os.getenv("MINICHAT_LANG", "en")
T = {
    "en": {"welcome": "Command line chat mode (type 'quit' to exit)", "you": "You: ", "ai": "AI: "},
    "zh": {"welcome": "命令行对话模式（输入 'quit' 退出）", "you": "你：", "ai": "AI："}
}
t = T[lang]

model, sp, device = load_model_and_tokenizer(lang=lang)

print(t["welcome"])
history = ""
while True:
    user_input = input("\n" + t["you"])
    if user_input.lower() == 'quit':
        break
    prompt = format_chat_prompt(user_input)
    full_prompt = history + prompt
    input_ids = torch.tensor([sp.bos_id()] + sp.encode(full_prompt, out_type=int), device=device).unsqueeze(0)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200)
    response_ids = output_ids[0, input_ids.shape[1]:].tolist()
    response = sp.decode(response_ids)
    if "### Assistant:" in response:
        response = response.split("### Assistant:")[-1].strip()
    print(t["ai"] + response)
    history += prompt + response + "\n"
    if len(history) > 1000:
        history = history[-1000:]