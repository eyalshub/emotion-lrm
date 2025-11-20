# src/data/build_cot_dataset.py

import json
import os
import random
import time
from typing import Dict, List

# Correct imports
from src.reasoning.prompt_templates import build_inference_prompt
from src.models.inference_ollama import (
    run_ollama,
    extract_final_answer,
    resolve_ollama_model
)

# ----------------------------------------------
# Configuration
# ----------------------------------------------

INPUT_TRAIN = "data/processed/goemotions_clean_train.jsonl"
INPUT_VAL   = "data/processed/goemotions_clean_val.jsonl"

OUTPUT_TRAIN = "data/processed/goemotions_cot_train.jsonl"
OUTPUT_VAL   = "data/processed/goemotions_cot_val.jsonl"

DEFAULT_MODEL = "phi"     # fastest model on local machine
MAX_RETRIES = 3           # retry attempts before skipping


# ----------------------------------------------
# Generate CoT for a single example
# ----------------------------------------------

def generate_cot_record(text: str, model_key: str) -> Dict | None:
    """
    Generates a single Chain-of-Thought record for one text example.
    Includes retry logic in case of Ollama timeouts or failures.
    """

    model_name = resolve_ollama_model(model_key)
    prompt = build_inference_prompt(text)

    for attempt in range(MAX_RETRIES):
        try:
            raw = run_ollama(prompt, model_name=model_name)
            emotion = extract_final_answer(raw)

            return {
                "text": text,
                "emotion": emotion,
                "cot": raw
            }

        except Exception as e:
            print(f"⚠️ Generation failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(1)

    print("❌ Skipped example due to repeated failures.")
    return None


# ----------------------------------------------
# Build full dataset
# ----------------------------------------------

def build_dataset(input_path: str, output_path: str, model_key: str, limit: int = None):
    """
    Loads a dataset file, generates CoT explanations for each example,
    and saves a new JSONL dataset enriched with:
      - text
      - gold emotional label
      - model's predicted emotion
      - full CoT reasoning (raw output)
    """

    print(f"📥 Loading file: {input_path}")

    lines = []
    with open(input_path, "r", encoding="utf8") as f:
        for line in f:
            lines.append(json.loads(line))

    # Optional: limit dataset size
    if limit:
        print(f"Using subset of {limit} examples")
        lines = random.sample(lines, limit)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = len(lines)
    print(f"🚀 Generating CoT for {total} examples using model '{model_key}'")

    with open(output_path, "w", encoding="utf8") as out:
        for idx, row in enumerate(lines):
            text = row["text"]
            label = row["label"]

            # Generate CoT record
            record = generate_cot_record(text, model_key=model_key)
            if not record:
                continue  # skip failures

            # Add the original gold label
            record["gold_label"] = label

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx % 20 == 0:
                print(f"{idx}/{total} done...")

    print(f"💾 Saved: {output_path}")


# ----------------------------------------------
# Main script entry point
# ----------------------------------------------

if __name__ == "__main__":
    LIMIT = 200   # start with 200 examples for a lightweight LoRA fine-tune

    build_dataset(INPUT_TRAIN, OUTPUT_TRAIN, model_key=DEFAULT_MODEL, limit=LIMIT)
    build_dataset(INPUT_VAL, OUTPUT_VAL, model_key=DEFAULT_MODEL, limit=int(LIMIT * 0.2))
