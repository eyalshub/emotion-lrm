# src/models/inference_lora.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.models.base_model_loader import load_tokenizer, load_base_model


# ---------------------------------------------------------
# 1. Load LoRA + base model
# ---------------------------------------------------------

def load_lora_model(model_key: str, lora_path: str, device: str = "cpu"):
    """
    Loads a base HF model + merges the LoRA adapter.
    Works fully on CPU.
    """

    # Load tokenizer
    tokenizer = load_tokenizer(model_key)

    # Load base model
    base_model = load_base_model(
        model_key=model_key,
        load_in_4bit=False,
        device=device
    )

    # Load LoRA adapter
    print(f"[Inference] Loading LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float32
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# 2. Build inference prompt
# ---------------------------------------------------------

def build_inference_prompt(text: str) -> str:
    """
    The inference prompt used for emotion reasoning.
    MUST match your fine-tuning prompt style.
    """
    return f"""
You are an emotional reasoning model.
Think step by step and explain the emotion behind the text.
At the end, output: Final Emotion: <emotion>

Text: "{text}"

=== Answer ===
"""


# ---------------------------------------------------------
# 3. Run inference
# ---------------------------------------------------------

def run_inference(model, tokenizer, text: str, device="cpu"):
    prompt = build_inference_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


# ---------------------------------------------------------
# 4. Extract the emotion label (simple)
# ---------------------------------------------------------

import re

def extract_emotion(output: str) -> str:
    """
    Extracts: Final Emotion: <emotion>
    """
    match = re.search(r"Final Emotion:\s*(.*)", output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "unknown"


# ---------------------------------------------------------
# 5. Main test
# ---------------------------------------------------------

if __name__ == "__main__":
    MODEL_KEY = "qwen0.5"
    LORA_PATH = "models/emotion-lrm-qwen0.5"

    text = "I'm so nervous about tomorrow."

    tokenizer, model = load_lora_model(MODEL_KEY, LORA_PATH)

    output = run_inference(model, tokenizer, text)
    print("\n=== RAW MODEL OUTPUT ===")
    print(output)

    emotion = extract_emotion(output)
    print("\n🎭 Predicted Emotion:", emotion)
