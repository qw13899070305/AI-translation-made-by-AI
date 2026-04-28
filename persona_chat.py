import torch
import os
from loader import load_model_and_tokenizer

lang = os.getenv("MINICHAT_LANG", "en")
model, sp, device = load_model_and_tokenizer(lang=lang)

persona_file = "persona.txt"
if os.path.exists(persona_file):
    with open(persona_file, "r", encoding="utf-8") as f:
        PERSONA = f.read().strip()
else:
    PERSONA = "You are a helpful assistant."

def persona_format_prompt(instruction):
    return f"{PERSONA}\n### User: {instruction}\n### Assistant: "

print(f"🎭 Persona Chat Mode (type 'quit' to exit)\nCurrent persona: {PERSONA[:50]}...")
history = ""
while True:
    user_input = input("You: " if lang == "en" else "你：")
    if user_input.lower() == 'quit':
        break
    prompt = persona_format_prompt(user_input)
    full_prompt = history + prompt
    input_ids = torch.tensor([sp.bos_id()] + sp.encode(full_prompt, out_type=int), device=device).unsqueeze(0)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200)
    response_ids = output_ids[0, input_ids.shape[1]:].tolist()
    response = sp.decode(response_ids)
    if "### Assistant:" in response:
        response = response.split("### Assistant:")[-1].strip()
    print(f"AI: {response}" if lang == "en" else f"AI：{response}")
    history += prompt + response + "\n"
    if len(history) > 1000:
        history = history[-1000:]