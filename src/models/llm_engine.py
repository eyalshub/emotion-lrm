# src/models/llm_engine.py

import threading
from typing import Optional

from src.models.inference_ollama import generate_emotion_reasoning_ollama
from src.models.base_model_loader import load_local_model
from src.models.inference import generate_emotion_reasoning


class LLMEngine:
    """
    Unified inference engine for EmotionLRM.
    Supports:
      - Ollama models
      - Local HuggingFace models (CPU/GPU)
      - Optional LoRA fine-tuned weights
      - Optional 4-bit loading for small hardware

    The engine lazily loads the local HF model only once,
    which dramatically improves performance.
    """

    _hf_model = None
    _hf_tokenizer = None
    _load_lock = threading.Lock()

    def __init__(
        self,
        use_ollama: bool = True,
        model_key: str = "phi",
        lora_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "cpu",
    ):
        self.use_ollama = use_ollama
        self.model_key = model_key
        self.lora_path = lora_path
        self.load_in_4bit = load_in_4bit
        self.device = device  # "cpu" or "cuda"

    # --------------------------
    # Public method
    # --------------------------

    def generate(self, text: str) -> str:
        """
        Main entry point. Runs emotion reasoning on the given text.
        Dispatches to Ollama or HF engine.
        """
        if self.use_ollama:
            return generate_emotion_reasoning_ollama(
                text=text,
                model_name=self.model_key
            )

        # HuggingFace engine
        return self._run_hf(text)

    # --------------------------
    # HuggingFace backend
    # --------------------------

    def _run_hf(self, text: str) -> str:
        """
        Runs generation using local HuggingFace model.
        Ensures the model/tokenizer are loaded once.
        """
        self._ensure_hf_loaded()

        return generate_emotion_reasoning(
            text=text,
            model=self._hf_model,
            tokenizer=self._hf_tokenizer,
            device=self.device
        )

    def _ensure_hf_loaded(self):
        """
        Loads HF model/tokenizer only once across the entire app.
        Thread-safe for FastAPI / Streamlit.
        """
        if self._hf_model is not None:
            return

        with self._load_lock:
            if self._hf_model is not None:
                return

            tokenizer, model = load_local_model(
                model_key=self.model_key,
                lora_path=self.lora_path,
                load_in_4bit=self.load_in_4bit,
                device=self.device,
            )

            self._hf_tokenizer = tokenizer
            self._hf_model = model

            print(f"[LLMEngine] Loaded local model '{self.model_key}' (LoRA={self.lora_path})")


# Convenience builder

def create_engine_ollama(model: str = "phi"):
    return LLMEngine(use_ollama=True, model_key=model)


def create_engine_local(
    model_key: str,
    lora_path: Optional[str] = None,
    load_in_4bit: bool = True,
    device: str = "cpu",
):
    return LLMEngine(
        use_ollama=False,
        model_key=model_key,
        lora_path=lora_path,
        load_in_4bit=load_in_4bit,
        device=device
    )
