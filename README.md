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

```mermaid
flowchart TD

    %% ------------------------
    %% LRM Definition
    %% ------------------------
    A[**What is an LRM?**<br><br>
      Lightweight Reasoning Model:<br>
      • Small model (Qwen/Llama)<br>
      • Fine-tuned with LoRA<br>
      • Learns reasoning from CoT<br>
      • Runs on CPU (Intel i7)<br>
      • Outputs emotion + explanation]:::title

    %% ------------------------
    %% RAW DATA → CLEANING
    %% ------------------------
    A --> B[**Raw Dataset**<br>
            GoEmotions (58k Reddit comments)<br>
            `data/raw/goemotions.csv`]:::dataset

    B --> C[**Advanced Cleaning Pipeline**<br>
            `src/data/preprocess_goemotions.py`<br><br>
            • Annotator agreement filtering<br>
            • Neutral-dominance reduction<br>
            • NLP token analysis<br>
            • Sentence-embedding outlier removal<br>
            • Remove spam / short / low-affect text]:::process

    C --> D[**Cleaned Dataset**<br>
            `data/processed/goemotions_clean.jsonl`]:::dataset

    %% ------------------------
    %% COT GENERATION
    %% ------------------------
    D --> E[**CoT Generation**<br>
            `src/data/build_cot_dataset.py`<br><br>
            Large LLM produces reasoning:<br>
            • Step-by-step emotional logic<br>
            • Template-guided CoT<br>
            • Heuristic validation<br>
            • Reject low-quality reasoning]:::reasoning

    E --> F[**CoT-Augmented Datasets**<br>
            `goemotions_cot_train.jsonl`<br>
            `goemotions_cot_val.jsonl`]:::dataset

    %% ------------------------
    %% TRAINING (LoRA)
    %% ------------------------
    F --> G[**LoRA Fine-Tuning**<br>
            `src/models/lora_finetune.py`<br>
            Configs: `configs/training_config.yaml`<br><br>
            • Qwen/Llama small as base<br>
            • PEFT + LoRA adapters<br>
            • CPU-friendly optimization<br>
            • Distills CoT reasoning]:::training

    G --> H[**LRM — Lightweight Reasoning Model**<br>
            (LoRA adapters stored under `models/lora/`)<br><br>
            • Compact<br>
            • Emotion-understanding<br>
            • CoT reasoning enabled]:::model

    %% ------------------------
    %% INFERENCE
    %% ------------------------
    H --> I[**Inference Engine**<br>
            `src/models/inference.py`<br><br>
            Input text → Emotion + CoT reasoning]:::inference

    %% ------------------------
    %% API + UI
    %% ------------------------
    I --> J[**FastAPI Endpoint (Optional)**<br>
            `src/api/app.py`<br>
            `/analyze_text`]:::api

    I --> K[**Streamlit UI (Optional)**<br>
            `src/ui/app_streamlit.py`<br>
            Interactive emotion + reasoning viewer]:::ui

    %% ------------------------
    %% EVALUATION
    %% ------------------------
    H --> L[**Evaluation Suite**<br>
            `src/reasoning/evaluation.py`<br><br>
            • Accuracy metrics<br>
            • Confusion matrix<br>
            • Reasoning-quality checks<br>
            • Outlier reasoning detection]:::eval

    %% ------------------------
    %% STYLES
    %% ------------------------
    classDef title fill:#1e1e1e,stroke:#777,color:#fff,font-weight:bold;
    classDef dataset fill:#2B4C7E,stroke:#1b2a44,color:#fff;
    classDef process fill:#406E8E,stroke:#1c3b50,color:#fff;
    classDef reasoning fill:#4C9F70,stroke:#2d6646,color:#fff;
    classDef training fill:#36827F,stroke:#235856,color:#fff;
    classDef model fill:#2E8C66,stroke:#1b523f,color:#fff;
    classDef inference fill:#5FA777,stroke:#3a6d4b,color:#fff;
    classDef api fill:#8EB95F,stroke:#587a3c,color:#fff;
    classDef ui fill:#A6C36F,stroke:#6e8447,color:#fff;
    classDef eval fill:#89A16A,stroke:#526240,color:#fff;
```

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




