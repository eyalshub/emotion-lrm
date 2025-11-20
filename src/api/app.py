# src/api/app.py

from fastapi import FastAPI
from pydantic import BaseModel

from src.models.llm_engine import create_engine_ollama, create_engine_local
from src.reasoning.emotion_reasoner import analyze_text


app = FastAPI(
    title="Emotion LRM API",
    description="Mini-LRM Framework (Ollama + HF engines)",
    version="2.1"
)


# -------------------------------------------------------
# Request Schema
# -------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str
    samples: int = 1               # default low for CPU
    engine: str = "ollama"         # "ollama" or "hf"
    model_key: str = "phi"         # ollama models: phi / llama3.2
    lora_path: str = None          # HF only
    load_in_4bit: bool = False     # HF only
    device: str = "cpu"            # HF: "cpu" or "cuda"


# -------------------------------------------------------
# Root Endpoint
# -------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "2.1",
        "available_engines": ["ollama", "hf"],
        "message": "Emotion LRM API is running!"
    }


# -------------------------------------------------------
# Main Endpoint: Analyze Emotion
# -------------------------------------------------------

@app.post("/analyze_text")
def analyze(req: AnalyzeRequest):

    # ---------------------
    # OLLAMA ENGINE
    # ---------------------
    if req.engine == "ollama":
        engine = create_engine_ollama(model=req.model_key)
        result = analyze_text(
            engine=engine,
            text=req.text,
            samples=req.samples
        )
        result["engine"] = "ollama"
        return result

    # ---------------------
    # HF ENGINE
    # ---------------------
    engine = create_engine_local(
        model_key=req.model_key,
        lora_path=req.lora_path,
        load_in_4bit=req.load_in_4bit,
        device=req.device
    )

    result = analyze_text(
        engine=engine,
        text=req.text,
        samples=req.samples
    )
    result["engine"] = "hf"
    return result
