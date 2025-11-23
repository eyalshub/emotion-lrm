# emotion-lrm
"A compact LoRA-fine-tuned LRM for emotion understanding with chain-of-thought reasoning."

## Overview

**Emotion-LRM** is a lightweight reasoning model designed for emotion classification using
step-by-step *Chain-of-Thought (CoT)* reasoning.  
The project focuses on training small, efficient language models (Qwen / Llama) using **LoRA**
fine-tuning, making it possible to achieve high-quality emotional reasoning on consumer hardware
such as an Intel i7 laptop.

The system includes a complete pipeline:
- Preparing and cleaning the GoEmotions dataset
- Generating reasoning-rich CoT examples using a larger external LLM
- Fine-tuning a compact model with PEFT/LoRA
- Running inference with emotion labels + reasoning explanations

Emotion-LRM aims to demonstrate how powerful reasoning capabilities can be distilled into small,
accessible models without requiring GPUs or large compute environments.


## Features

- **🔍 Full Emotion Classification Pipeline**  
  From raw GoEmotions → cleaned data → CoT-enriched dataset → fine-tuned LRM.

- **🧠 Chain-of-Thought Augmentation**  
  Automatically generates reasoning steps using a larger external LLM to improve small-model performance.

- **⚙️ Lightweight LoRA Fine-Tuning**  
  Optimized training pipeline that runs efficiently on CPU machines (Intel i7) without requiring GPU.

- **🤖 Inference with Reasoning**  
  The model outputs both the predicted emotion *and* the reasoning behind the prediction.

- **📊 Evaluation Tools**  
  Includes accuracy reports, confusion matrix, and CoT quality checks.

- **📁 Modular Codebase**  
  Clean separation of data processing, model loading, training, evaluation, and inference.

- **🧩 Debug-Friendly**  
  Includes tiny sample datasets for rapid iteration and CPU-only testing.

- **🚀 HuggingFace Integration**  
  Works seamlessly with Qwen, Llama, and other transformer architectures.


## LRM Architecture & Full Pipeline

flowchart TD

    A["**What is an LRM?**<br/>
    Lightweight Reasoning Model:<br/>
    - Small model (Qwen/Llama)<br/>
    - LoRA fine-tuned<br/>
    - Learns CoT reasoning<br/>
    - Runs on CPU<br/>
    - Outputs emotion + explanation"] --> B

    B["**Raw Dataset**<br/>
    GoEmotions (58k)<br/>
    data/raw/goemotions.csv"] --> C

    C["**Cleaning Pipeline**<br/>
    preprocess_goemotions.py<br/>
    - Annotator filtering<br/>
    - Neutral reduction<br/>
    - NLP token checks<br/>
    - Embedding outliers<br/>
    - Remove spam/short text"] --> D

    D["**Cleaned Dataset**<br/>
    goemotions_clean.jsonl"] --> E

    E["**CoT Generation**<br/>
    build_cot_dataset.py<br/>
    - Generate reasoning<br/>
    - Template CoT<br/>
    - Heuristic filtering"] --> F

    F["**CoT-Augmented Data**<br/>
    cot_train.jsonl<br/>
    cot_val.jsonl"] --> G

    G["**LoRA Fine-Tuning**<br/>
    lora_finetune.py<br/>
    training_config.yaml<br/>
    - Qwen/Llama base<br/>
    - PEFT + LoRA<br/>
    - CPU-friendly"] --> H

    H["**LRM Model**<br/>
    models/lora/<br/>
    - Compact model<br/>
    - CoT reasoning"] --> I

    I["**Inference**<br/>
    inference.py<br/>
    text → emotion + reasoning"] --> J

    I --> K

    J["**FastAPI (optional)**<br/>
    api/app.py<br/>
    /analyze_text"]

    K["**Streamlit UI (optional)**<br/>
    ui/app_streamlit.py"]

    H --> L

    L["**Evaluation**<br/>
    evaluation.py<br/>
    - Accuracy<br/>
    - Confusion matrix<br/>
    - CoT quality"]




## Project Structure
```
emotion-lrm/
├── data/
│ ├── raw/
│ │ └── goemotions.csv # Raw GoEmotions dataset
│ ├── processed/
│ │ ├── goemotions_clean.jsonl # Cleaned & filtered dataset
│ │ ├── goemotions_cot_train.jsonl # CoT-augmented training set
│ │ └── goemotions_cot_val.jsonl # CoT-augmented validation set
│ └── samples/
│ └── tiny_debug_1k.jsonl # Small subset for debugging on CPU/i7
│
├── src/
│ ├── data/
│ │ ├── preprocess_goemotions.py # Cleaning, balancing, splitting
│ │ └── build_cot_dataset.py # Automatic CoT generation using a larger LLM
│ │
│ ├── models/
│ │ ├── base_model_loader.py # Load Qwen/Llama base model
│ │ ├── lora_finetune.py # LoRA fine-tuning pipeline
│ │ └── inference.py # Run inference with LoRA (CPU/GPU)
│ │
│ ├── reasoning/
│ │ ├── prompt_templates.py # Templates for reasoning prompts
│ │ ├── emotion_reasoner.py # Core reasoning logic (CoT generation/analysis)
│ │ └── evaluation.py # Metrics, confusion matrix, reasoning-quality checks
│ │
│ ├── api/
│ │ └── app.py # Minimal FastAPI/Flask API for /analyze_text
│ │
│ └── ui/
│ └── app_streamlit.py # Streamlit demo UI (text → emotion + reasoning)
│
├── notebooks/
│ ├── 01_explore_goemotions.ipynb # Data exploration
│ ├── 02_generate_cot_with_big_lrm.ipynb # CoT generation (Colab recommended)
│ └── 03_lora_finetune_emotion_lrm.ipynb # Training notebook
│
├── tests/
│ ├── test_data_pipeline.py # Tests for preprocessing + CoT generation
│ ├── test_inference_pipeline.py # Tests for inference stability & outputs
│ └── test_reasoning_quality.py # Tests for CoT consistency
│
├── configs/
│ ├── training_config.yaml # Hyperparameters, LoRA settings
│ └── model_config.yaml # Base model, context window, reasoning tokens
│
├── README.md
└── requirements.txt
```

## Dataset

The project is based on the **GoEmotions** dataset (27 emotions + neutral), originally released
by Google. The dataset includes 58k Reddit comments labeled by multiple human annotators.

### 🧹 Data Cleaning & Label Filtering

A custom cleaning pipeline was applied to improve the emotional signal and reduce noise:

- **Annotator Agreement Filtering**  
  Samples were filtered based on the *percentage of annotators* who selected each label.  
  Low-confidence labels (e.g., multiple annotators marking “neutral”) were removed to improve clarity.

- **Neutral-Dominant Removal**  
  Entries where most annotators assigned the label **neutral** (or near-neutral) were down-weighted
  or removed to avoid bias toward generic emotional predictions.

- **Text Quality Checks**  
  Removed:
  - very short texts  
  - URLs / spam-like content  
  - duplicates  
  - comments with no emotional information

### 🔬 NLP-Based Cleaning

To improve dataset purity, several NLP techniques were used:

- **Text embeddings (SentenceTransformers)**  
  Used for:
  - detecting outlier samples  
  - identifying inconsistencies between label and semantic meaning  
  - grouping semantically similar samples for filtering

- **Token-level linguistic analysis**  
  Applied to detect:
  - purely factual statements  
  - low-affect sentences  
  - sarcastic or highly ambiguous samples

Together, these methods help create a **more concise, emotionally consistent dataset**.

### 🧠 CoT Augmentation (Chain-of-Thought)

After cleaning, each example is enriched with a reasoning chain:

- A large external LLM generates a *step-by-step emotional explanation*
- Only high-quality reasoning chains (validated by templates & heuristics) are included
- Output is stored as:


## Usage

This section describes how to run the full Emotion-LRM pipeline:
1. Preprocess the dataset  
2. Generate Chain-of-Thought examples  
3. Fine-tune the compact model (LoRA)  
4. Run inference  
5. Evaluate model performance  

Make sure you have activated your virtual environment and installed all dependencies.


---

### 1️⃣ Preprocess the GoEmotions Dataset

Cleans the raw data, filters labels using agreement thresholds, removes noisy samples, and applies NLP/embedding-based quality checks.

```bash
python -m src.data.preprocess_goemotions
```
### 2️⃣ Generate Chain-of-Thought (CoT) Reasoning
```
python -m src.data.build_cot_dataset
```
### 3️⃣ Fine-Tune the Model Using LoRA
```
python -m src.models.lora_finetune
```
### 4️⃣ Run Inference
```
python -m src.models.inference "I feel disappointed and ignored..."
```

### 5️⃣ Evaluate the Model
```
python -m src.reasoning.evaluation
```

## Installation

Follow the steps below to set up the Emotion-LRM environment.

### 1. Clone the repository

```bash
git clone https://github.com/eyalshub/emotion-lrm.git
cd emotion-lrm

### Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

### macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

huggingface-cli login         # optional




