import gradio as gr
import torch
import os
from config import Config
from loader import load_model_and_tokenizer
from utils import format_chat_prompt
from rag_module import RAGModule
from vision_module import VisionEncoder

LANG = os.getenv("MINICHAT_LANG", "en")
T = {
    "title": {"en": "🤖 My Own AI Assistant", "zh": "🤖 我的专属 AI 助手"},
    "desc": {"en": "Integrated Hybrid Attention (Gated DeltaNet/CSA) + MoE + MTP", "zh": "集成混合注意力(Gated DeltaNet/CSA)+MoE+MTP"},
    "voice": {"en": "🎤 Voice Input", "zh": "🎤 语音输入"},
    "files": {"en": "Upload Documents (PDF/TXT)", "zh": "上传文档 (PDF/TXT)"},
    "image": {"en": "Upload Image", "zh": "上传图片"},
}

model, sp, device = load_model_and_tokenizer(lang=LANG)
cfg = Config()
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
        message = f"Based on the following information:\n{context}\n\nQuestion: {message}"
    conv += format_chat_prompt(message)
    input_ids = torch.tensor([sp.bos_id()] + sp.encode(conv, out_type=int), device=device).unsqueeze(0)
    vision_embeds = None
    if image is not None and vision_encoder is not None:
        vision_embeds = vision_encoder([image]).to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=300, vision_embeds=vision_embeds)
    response_ids = output_ids[0, input_ids.shape[1]:].tolist()
    response = sp.decode(response_ids)
    if "### Assistant:" in response:
        response = response.split("### Assistant:")[-1].strip()
    return response

demo = gr.ChatInterface(
    fn=respond,
    title=T["title"][LANG],
    description=T["desc"][LANG],
    additional_inputs=[
        gr.Audio(source="microphone", type="numpy", label=T["voice"][LANG]),
        gr.File(file_count="multiple", label=T["files"][LANG]),
        gr.Image(type="pil", label=T["image"][LANG])
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=False)