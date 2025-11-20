# src/reasoning/emotion_reasoner_ollama.py

from statistics import mode, StatisticsError
from typing import List, Dict

from src.models.inference_ollama import generate_emotion_reasoning_ollama

DEFAULT_SAMPLES = 3


def aggregate_emotions(samples: List[Dict]) -> str:
    emotions = [s["emotion"] for s in samples if s["emotion"]]
    if not emotions:
        return ""
    try:
        return mode(emotions)
    except StatisticsError:
        return emotions[0]


def consistency_score(samples: List[Dict], final_emotion: str) -> float:
    if not final_emotion:
        return 0.0
    total = len(samples)
    agree = sum(1 for s in samples if s["emotion"] == final_emotion)
    return agree / total if total else 0.0


def choose_best_reasoning(samples: List[Dict], final_emotion: str) -> str:
    filtered = [s for s in samples if s["emotion"] == final_emotion]
    if not filtered:
        return samples[0]["reasoning"]
    return filtered[0]["reasoning"]


def analyze_text_ollama(
    text: str,
    samples: int = DEFAULT_SAMPLES,
    model_key: str = "phi"
) -> Dict:

    all_outputs = []

    for _ in range(samples):
        out = generate_emotion_reasoning_ollama(
            text=text,
            model_key=model_key     # ✔️ תיקון!
        )
        all_outputs.append(out)

    final_emotion = aggregate_emotions(all_outputs)
    score = consistency_score(all_outputs, final_emotion)
    final_reasoning = choose_best_reasoning(all_outputs, final_emotion)

    return {
        "text": text,
        "emotion": final_emotion,
        "consistency_score": score,
        "reasoning": final_reasoning,
        "samples": all_outputs,
        "engine": "ollama"
    }


if __name__ == "__main__":
    test = "I feel so empty and tired of everything."
    print(analyze_text_ollama(test))
