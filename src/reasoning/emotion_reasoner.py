# src/reasoning/emotion_reasoner.py

from typing import List, Dict
from statistics import mode, StatisticsError

from src.models.llm_engine import LLMEngine


DEFAULT_SAMPLES = 1


# -------------------------------------------------------
# Majority Vote
# -------------------------------------------------------

def aggregate_emotions(samples: List[Dict]) -> str:
    """
    Returns the majority-voted emotion from the samples list.
    """
    emotions = [s.get("emotion") for s in samples if s.get("emotion")]

    if not emotions:
        return ""

    try:
        return mode(emotions)
    except StatisticsError:
        return emotions[0]  # fallback if no clear majority


# -------------------------------------------------------
# Consistency Score
# -------------------------------------------------------

def consistency_score(samples: List[Dict], final_emotion: str) -> float:
    """
    Measures how many samples agree with the final emotion.
    """
    if not final_emotion:
        return 0.0

    total = len(samples)
    agree = sum(1 for s in samples if s.get("emotion") == final_emotion)

    return agree / total if total > 0 else 0.0


# -------------------------------------------------------
# Choose best reasoning
# -------------------------------------------------------

def choose_best_reasoning(samples: List[Dict], final_emotion: str) -> str:
    """
    Selects reasoning from the sample that matches the final emotion.
    """
    filtered = [s for s in samples if s.get("emotion") == final_emotion]

    if filtered:
        return filtered[0].get("reasoning", "")

    # fallback: take the first reasoning
    return samples[0].get("reasoning", "")


# -------------------------------------------------------
# Main Agent Logic (HF or Ollama via LLMEngine)
# -------------------------------------------------------

def analyze_text(
    engine: LLMEngine,
    text: str,
    samples: int = DEFAULT_SAMPLES,
) -> Dict:
    """
    Runs the LLM engine multiple times, aggregates the results,
    and produces a final emotional classification + reasoning.

    engine: LLMEngine (HF or Ollama)
    text: input text
    samples: number of repeated CoT generations
    """

    all_outputs = []

    for _ in range(samples):
        out = engine.generate(text)
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
    }


# -------------------------------------------------------
# Manual test
# -------------------------------------------------------

if __name__ == "__main__":
    from src.models.llm_engine import create_engine_ollama

    engine = create_engine_ollama("phi")

    result = analyze_text(
        engine=engine,
        text="I really miss her... everything feels empty.",
        samples=3
    )

    print(result)
