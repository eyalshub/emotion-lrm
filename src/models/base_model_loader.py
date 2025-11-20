# src/models/base_model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Tuple, Optional
import os


# ----------------------------------------------------------
# Mapping short keys → HuggingFace model names
# ----------------------------------------------------------

HF_MODEL_MAP = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama3.2": "meta-llama/Llama-3.2-1B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",

    # ✅ Correct Qwen 0.5B model
    "qwen0.5": "Qwen/Qwen2.5-0.5B",
}


# ----------------------------------------------------------
# Tokenizer Loader
# ----------------------------------------------------------

def load_tokenizer(model_key: str):
    if model_key not in HF_MODEL_MAP:
        raise ValueError(f"Unknown HF model key: {model_key}")

    model_name = HF_MODEL_MAP[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ----------------------------------------------------------
# Base Model Loader
# ----------------------------------------------------------

def load_base_model(
    model_key: str,
    load_in_4bit: bool = False,
    device: str = "cpu"
):
    if model_key not in HF_MODEL_MAP:
        raise ValueError(f"Unknown HF model key: {model_key}")

    model_name = HF_MODEL_MAP[model_key]
    print(f"[HF] Loading base model: {model_name}")

    quant_config = None

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("[HF] Using 4-bit quantization")
        except Exception:
            print("⚠️ 4-bit quantization not supported on Windows. Using fp32 instead.")
            quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.float32,
        device_map=None,
    )

    return model


# ----------------------------------------------------------
# LoRA Loader
# ----------------------------------------------------------

def load_model_with_lora(
    model_key: str,
    lora_path: str,
    load_in_4bit: bool = False,
    device: str = "cpu"
):
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    print(f"[HF] Loading base model + LoRA: {lora_path}")

    model = load_base_model(model_key, load_in_4bit=load_in_4bit)
    model = PeftModel.from_pretrained(model, lora_path)

    return model


# ----------------------------------------------------------
# Unified loader used everywhere
# ----------------------------------------------------------

def load_local_model(
    model_key: str,
    lora_path: Optional[str] = None,
    load_in_4bit: bool = False,
    device: str = "cpu"
):
    tokenizer = load_tokenizer(model_key)

    if lora_path:
        model = load_model_with_lora(
            model_key=model_key,
            lora_path=lora_path,
            load_in_4bit=load_in_4bit,
            device=device
        )
    else:
        model = load_base_model(
            model_key=model_key,
            load_in_4bit=load_in_4bit,
            device=device
        )

    return tokenizer, model