# src/models/inference.py

import re
import torch
from typing import Dict, Tuple

from src.reasoning.prompt_templates import build_inference_prompt



VALID_EMOTIONS = {
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
}

# -------------------------------------------------------
# 1. Text Generation (HF local)
# -------------------------------------------------------

def run_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.4,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: str = "cpu",
) -> str:
    """
    Runs text generation using a local HuggingFace model.
    Assumes model & tokenizer are already loaded by LLMEngine.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


# -------------------------------------------------------
# 2. Extracting the final emotion from the output
# -------------------------------------------------------


def extract_final_answer(text: str) -> str:
    """
    Extracts emotion from: Final Answer: <emotion>
    But ONLY returns it if it's in the GoEmotions label set.
    """
    pattern = r"(Final\s+(Answer|Emotion)\s*:\s*)([a-zA-Z]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""

    candidate = match.group(3).lower().strip()

    return candidate if candidate in VALID_EMOTIONS else ""


# -------------------------------------------------------
# 3. Extract reasoning (text before Final Answer)
# -------------------------------------------------------

def extract_reasoning(text: str) -> str:
    """
    Extracts reasoning before the 'Final Answer:' section.
    """
    parts = re.split(r"Final\s+(Answer|Emotion)\s*:", text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        return parts[0].strip()
    return text.strip()


# -------------------------------------------------------
# 4. Main inference function (HF backend)
# -------------------------------------------------------

def generate_emotion_reasoning(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
) -> Dict:
    """
    Runs emotion reasoning using a loaded HF model.
    This function assumes:
        - The model is already loaded (LLMEngine handles loading)
        - The tokenizer is already loaded
        - Device is defined ("cpu"/"cuda")

    Returns a dict:
        {
            "emotion": <str>,
            "reasoning": <str>,
            "raw_output": <str>
        }
    """

    # Build final prompt
    prompt = build_inference_prompt(text)

    # Generate raw output
    raw_output = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
    )

    # Extract final fields
    emotion = extract_final_answer(raw_output)
    reasoning = extract_reasoning(raw_output)

    return {
        "emotion": emotion,
        "reasoning": reasoning,
        "raw_output": raw_output
    }


# -------------------------------------------------------
# optional quick test
# -------------------------------------------------------

if __name__ == "__main__":
    print("This file is intended to be used through LLMEngine, not standalone.")
