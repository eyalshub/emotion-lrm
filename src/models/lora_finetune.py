# src/models/lora_finetune.py

import os
from dataclasses import dataclass
from typing import Dict, Any

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

from src.models.base_model_loader import load_local_model
from src.reasoning.prompt_templates import build_training_prompt


# ============================================================
# CONFIG
# ============================================================

@dataclass
class FinetuneConfig:
    base_model_key: str = "qwen0.5"   # MUST exist in HF_MODEL_MAP
    train_file: str = "data/processed/goemotions_cot_train.jsonl"
    val_file: str = "data/processed/goemotions_cot_val.jsonl"
    output_dir: str = "models/emotion-lrm-qwen0.5"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 2e-4
    max_seq_length: int = 256
    gradient_accumulation_steps: int = 4
    use_4bit: bool = False  # Windows cannot use bitsandbytes
    device: str = "cpu"     # FORCE CPU for Windows


# ============================================================
# LOAD DATASET
# ============================================================

def load_datasets(cfg: FinetuneConfig):
    data_files = {
        "train": cfg.train_file,
        "validation": cfg.val_file,
    }
    return load_dataset("json", data_files=data_files)


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_dataset(tokenizer, raw_datasets, cfg: FinetuneConfig):

    def _build_example(example: Dict[str, Any]):
        return {
            "text": build_training_prompt(
                text=example["text"],
                cot=example["cot"],
                label=example["gold_label"]
            )
        }

    processed = raw_datasets.map(
        _build_example,
        remove_columns=raw_datasets["train"].column_names
    )

    def _tokenize(example):
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
        )
        # THE KEY FIX:
        enc["labels"] = enc["input_ids"].copy()
        return enc

    return processed.map(_tokenize, batched=True)

# ============================================================
# LoRA SETUP
# ============================================================

def create_lora_model(base_model, r=8, alpha=16, dropout=0.05):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, config)


# ============================================================
# TRAINING LOOP
# ============================================================

def run_finetuning(cfg: FinetuneConfig):

    # Load tokenizer + base model (no 4bit!)
    tokenizer, base_model = load_local_model(
        model_key=cfg.base_model_key,
        load_in_4bit=False,
        device=cfg.device
    )

    # pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    raw = load_datasets(cfg)
    tokenized = prepare_dataset(tokenizer, raw, cfg)

    # LoRA
    model = create_lora_model(base_model)

    # Collator best for SFT
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

    # Training settings
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=False,
        bf16=False,
        report_to="none",
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    # Save model + tokenizer + LoRA adapter
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cfg = FinetuneConfig()
    run_finetuning(cfg)
