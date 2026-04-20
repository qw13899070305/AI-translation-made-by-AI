import gradio as gr
import torch
import sentencepiece as spm
from config import Config
from model import MiniChat
from utils import format_chat_prompt
from rag_module import RAGModule
from vision_module import VisionEncoder
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

rag = RAGModule()
vision_encoder = VisionEncoder().to(device) if cfg.use_multimodal else None

def respond(message, history, uploaded_files, image):
    conv = ""
    for h in history:
        conv += format_chat_prompt(h[0], h[1]) + "\n"
    if uploaded_files:
        rag.add_documents([f.name for f in uploaded_files])
    retrieved = rag.retrieve(message, k=2)
    context = "\n".join(retrieved)
    if context:
        message = f"根据以下信息回答问题：\n{context}\n\n问题：{message}"
    conv += format_chat_prompt(message)
    input_ids = torch.tensor([sp.bos_id()] + sp.encode(conv, out_type=int), device=device).unsqueeze(0)
    vision_embeds = None
    if image is not None and vision_encoder is not None:
        vision_embeds = vision_encoder([image])
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=300, vision_embeds=vision_embeds)
    response_ids = output_ids[0, input_ids.shape[1]:].tolist()
    response = sp.decode(response_ids)
    if "### 助手:" in response:
        response = response.split("### 助手:")[-1].strip()
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="🤖 全能型自研 AI 助手",
    description="集成了 LoRA 微调、RAG 检索、GQA/MoE 架构、多模态理解。",
    additional_inputs=[
        gr.File(file_count="multiple", label="上传文档 (PDF/TXT) 用于 RAG"),
        gr.Image(type="pil", label="上传图片 (多模态)")
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=False)