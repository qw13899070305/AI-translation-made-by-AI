import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import uvicorn
from loader import load_model_and_tokenizer
from utils import format_chat_prompt

lang = os.getenv("MINICHAT_LANG", "en")
model, sp, device = load_model_and_tokenizer(lang=lang)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        prompt = format_chat_prompt(request.message)
        input_ids = torch.tensor([sp.bos_id()] + sp.encode(prompt, out_type=int), device=device).unsqueeze(0)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=200)
        response_ids = output_ids[0, input_ids.shape[1]:].tolist()
        response = sp.decode(response_ids)
        if "### Assistant:" in response:
            response = response.split("### Assistant:")[-1].strip()
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)