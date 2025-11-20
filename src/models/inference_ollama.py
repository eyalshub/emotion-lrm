# src/models/inference_ollama.py

import re
from typing import Dict

from src.models.ollama_client import ollama_generate
from src.reasoning.prompt_templates import build_inference_prompt


# -------------------------------------------------------
# 1. Model mapping (short key → Ollama model tag)
# -------------------------------------------------------

DEFAULT_OLLAMA_MODEL_KEY = "phi"

OLLAMA_MODEL_MAP = {
    "phi": "phi:latest",
    "phi3": "phi3:latest",
    "llama3.2": "llama3.2:latest",
    "qwen": "qwen:latest",
}


def resolve_ollama_model(model_key: str) -> str:
    """Converts short model key to an Ollama model tag."""
    return OLLAMA_MODEL_MAP.get(model_key, model_key)


# -------------------------------------------------------
# 2. Run Ollama backend
# -------------------------------------------------------

def run_ollama(prompt: str, model_name: str, debug: bool = False) -> str:
    """
    Runs an Ollama model and returns raw text output.
    Handles runtime errors gracefully.
    """

    if debug:
        print(f"\n🔥 [Ollama] Running model: {model_name}")
        print("📝 Prompt:")
        print(prompt)

    try:
        response = ollama_generate(
            model=model_name,
            prompt=prompt,
        )
    except Exception as e:
        return f"ERROR: Ollama runtime failure → {e}"

    return response


# -------------------------------------------------------
# 3. Emotion extraction (same regex as HF version)
# -------------------------------------------------------

def extract_final_answer(text: str) -> str:
    """
    Extracts:  Final Answer: <emotion>
    (or Final Emotion)
    """
    pattern = r"(Final\s+(Answer|Emotion)\s*:\s*)([a-zA-Z]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(3).lower().strip()
    return ""


def extract_reasoning(text: str) -> str:
    """Extracts everything before 'Final Answer' / 'Final Emotion'."""
    parts = re.split(r"Final\s+(Answer|Emotion)\s*:", text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        return parts[0].strip()
    return text.strip()


# -------------------------------------------------------
# 4. Full inference pipeline (Ollama backend)
# -------------------------------------------------------

def generate_emotion_reasoning_ollama(
    text: str,
    model_name: str = DEFAULT_OLLAMA_MODEL_KEY,
    debug: bool = False,
) -> Dict:
    """
    High-level emotional reasoning using an Ollama model.
    Returns:
        {
            "engine": "ollama",
            "model_key": ...,
            "model_used": ...,
            "emotion": ...,
            "reasoning": ...,
            "raw_output": ...
        }
    """

    # Resolve model tag
    model_used = resolve_ollama_model(model_name)

    # Build inference prompt
    prompt = build_inference_prompt(text)

    # Run Ollama inference
    raw_output = run_ollama(prompt, model_used, debug=debug)

    # If failed:
    if raw_output.startswith("ERROR:"):
        return {
            "engine": "ollama",
            "model_key": model_name,
            "model_used": model_used,
            "emotion": "",
            "reasoning": "Ollama error during generation.",
            "raw_output": raw_output,
        }

    # Extract emotion + reasoning
    emotion = extract_final_answer(raw_output)
    reasoning = extract_reasoning(raw_output)

    return {
        "engine": "ollama",
        "model_key": model_name,
        "model_used": model_used,
        "emotion": emotion,
        "reasoning": reasoning,
        "raw_output": raw_output,
    }


# -------------------------------------------------------
# Manual test
# -------------------------------------------------------

if __name__ == "__main__":
    example = "I feel exhausted and lonely today."
    out = generate_emotion_reasoning_ollama(example, model_name="phi", debug=True)
    print(out)
