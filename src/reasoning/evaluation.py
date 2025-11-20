# src/reasoning/evaluation.py

"""
Evaluation module for EmotionLRM.
Provides:
 - Dataset loading (GoEmotions / CoT dataset)
 - Running inference on a batch of examples
 - Computing Accuracy, Macro-F1, Micro-F1, Per-class F1
 - Saving predictions to a JSONL file
 - Optional reasoning quality scoring
"""

import json
import jsonlines
from typing import List, Dict, Callable

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from src.models.llm_engine import LLMEngine
from src.reasoning.prompt_templates import build_inference_prompt


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    """
    Loads JSONL dataset from disk.
    Each line should contain:
        {
            "text": "...",
            "label": "anger",
            "cot": "...",   # optional
        }
    """
    data = []
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            data.append(row)
    return data


# ---------------------------------------------------------
# 2. Run evaluation loop
# ---------------------------------------------------------

def evaluate_model(
    engine: LLMEngine,
    dataset: List[Dict],
    max_items: int = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Runs inference over the dataset and returns a list of prediction objects:
    {
        "text": ...,
        "gold": ...,
        "pred": ...,
        "reasoning": ...,
        "raw_output": ...
    }
    """

    results = []

    if max_items:
        dataset = dataset[:max_items]

    for i, sample in enumerate(dataset):
        text = sample["text"]
        gold = sample.get("label", "")

        if verbose:
            print(f"\n[{i+1}/{len(dataset)}] Text: {text}")

        out = engine.generate(text)

        pred = out.get("emotion", "")
        reasoning = out.get("reasoning", "")
        raw = out.get("raw_output", "")

        results.append({
            "text": text,
            "gold": gold,
            "pred": pred,
            "reasoning": reasoning,
            "raw_output": raw
        })

        if verbose:
            print(f" → gold: {gold}, pred: {pred}")

    return results


# ---------------------------------------------------------
# 3. Compute metrics
# ---------------------------------------------------------

def compute_metrics(results: List[Dict]) -> Dict:
    """
    Computes accuracy, macro/micro F1, and per-class F1.
    """

    gold_labels = [r["gold"] for r in results]
    pred_labels = [r["pred"] for r in results]

    # Basic metrics
    accuracy = accuracy_score(gold_labels, pred_labels)
    macro_f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
    micro_f1 = f1_score(gold_labels, pred_labels, average="micro", zero_division=0)

    # Detailed per-label breakdown
    report = classification_report(
        gold_labels,
        pred_labels,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_label_f1": report
    }


# ---------------------------------------------------------
# 4. Save predictions to disk
# ---------------------------------------------------------

def save_predictions_jsonl(results: List[Dict], path: str):
    """Saves evaluation results as JSONL."""
    with jsonlines.open(path, "w") as writer:
        for row in results:
            writer.write(row)


# ---------------------------------------------------------
# 5. Full evaluation pipeline helper
# ---------------------------------------------------------

def run_full_evaluation(
    engine: LLMEngine,
    dataset_path: str,
    output_path: str = None,
    max_items: int = None,
    verbose: bool = True,
) -> Dict:
    """
    High-level helper:
     - Loads dataset
     - Runs inference
     - Computes metrics
     - Saves predictions (optional)
    """

    dataset = load_jsonl(dataset_path)

    results = evaluate_model(
        engine=engine,
        dataset=dataset,
        max_items=max_items,
        verbose=verbose
    )

    metrics = compute_metrics(results)

    if verbose:
        print("\n===== METRICS =====")
        print(json.dumps(metrics, indent=2))

    if output_path:
        save_predictions_jsonl(results, output_path)
        if verbose:
            print(f"\nSaved predictions to: {output_path}")

    return metrics


# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------

if __name__ == "__main__":
    # Example: evaluate on the validation CoT dataset
    engine = LLMEngine(
        use_ollama=False,           # or True
        model_key="phi",
        lora_path=None,
        device="cpu"
    )

    metrics = run_full_evaluation(
        engine=engine,
        dataset_path="data/processed/goemotions_cot_val.jsonl",
        output_path="evaluation_results.jsonl",
        max_items=10,    # test run
        verbose=True
    )
