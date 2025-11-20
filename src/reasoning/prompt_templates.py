# src/reasoning/prompt_templates.py

"""
Prompt templates for emotion reasoning, CoT generation, and training.
This module centralizes all text templates used for both training and inference.
"""

# -----------------------------
# System prompts
# -----------------------------

SYSTEM_EMOTION_ANALYSIS = """
You are EmotionLRM, a compact emotional reasoning model.
Your job is to analyze emotional content deeply, logically, and precisely.
Always think step-by-step, ensure internal consistency, and identify the final emotion.
"""

SYSTEM_COT_GENERATION = """
You are a large reasoning model used to generate high-quality Chain-of-Thought explanations
for emotional classification tasks. Your answers will be used for training a smaller LRM model.
Your job is to produce clear, structured reasoning followed by a final emotion label.
"""


# -----------------------------
# Templates for inference
# -----------------------------

INFERENCE_PROMPT_TEMPLATE = """
You are an expert emotional reasoning AI.
Analyze the emotional content of the text step by step.
Explain your reasoning clearly and logically.
At the end, write exactly: "Final Emotion: <emotion>".

Text: "{text}"

=== Answer ===
"""


def build_inference_prompt(text: str) -> str:
    """
    Build the prompt used during inference (runtime reasoning).
    """
    return INFERENCE_PROMPT_TEMPLATE.format(text=text.strip())


# -----------------------------
# Templates for training (with CoT)
# -----------------------------

TRAINING_TEMPLATE = """
### Instruction:
Analyze the emotional content of the text using detailed logical reasoning.
Then output a final label.

### Input Text:
{text}

### Chain-of-Thought:
{cot}

### Final Emotion:
{label}
"""


def build_training_prompt(text: str, cot: str, label: str) -> str:
    """
    Build supervised fine-tuning prompt for CoT training.
    """
    cot_clean = cot.strip() if cot else ""
    label_clean = label.strip() if label else ""
    text_clean = text.strip()

    return TRAINING_TEMPLATE.format(
        text=text_clean,
        cot=cot_clean,
        label=label_clean,
    )


# -----------------------------
# Templates for generating CoT with a large model
# -----------------------------

COT_GENERATION_TEMPLATE = """
You will analyze an input text and generate a high-quality chain-of-thought
explaining the emotional state behind the text.

Steps:
1. Identify emotional cues.
2. Explain your reasoning step-by-step.
3. Output the final emotion in the format: Final Emotion: <emotion>

Text: "{text}"
"""


def build_cot_generation_prompt(text: str) -> str:
    """
    Build prompt for generating CoT from a large model (GPT/Llama).
    """
    return COT_GENERATION_TEMPLATE.format(text=text.strip())
