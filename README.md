# emotion-lrm
"A compact LoRA-fine-tuned LRM for emotion understanding with chain-of-thought reasoning."


emotion-lrm/
├── data/
│   ├── raw/
│   │   └── goemotions.csv              # raw dataset
│   ├── processed/
│   │   ├── goemotions_clean.jsonl      # cleaned/filtered dataset
│   │   ├── goemotions_cot_train.jsonl  # with reasoning chains (train)
│   │   └── goemotions_cot_val.jsonl    # with reasoning chains (validation)
│   └── samples/
│       └── tiny_debug_1k.jsonl         # tiny subset for i7 debugging
│
├── src/
│   ├── data/
│   │   ├── preprocess_goemotions.py    # cleaning, balancing, train/val split
│   │   └── build_cot_dataset.py        # generate CoT via a large LRM
│   ├── models/
│   │   ├── base_model_loader.py        # load base model (Llama/Qwen)
│   │   ├── lora_finetune.py            # LoRA fine-tuning
│   │   └── inference.py                # run inference with the fine-tuned model (CPU/GPU)
│   ├── reasoning/
│   │   ├── prompt_templates.py         # CoT templates for emotional/logical reasoning
│   │   ├── emotion_reasoner.py         # reasoning logic (chain-of-thought)
│   │   └── evaluation.py               # performance metrics + reasoning quality analysis
│   ├── api/
│   │   └── app.py                      # FastAPI/Flask endpoint: /analyze_text
│   └── ui/
│       └── app_streamlit.py            # small demo UI (text → emotion + explanation)
│
├── notebooks/
│   ├── 01_explore_goemotions.ipynb
│   ├── 02_generate_cot_with_big_lrm.ipynb   # runs well on Colab
│   └── 03_lora_finetune_emotion_lrm.ipynb   # Colab/local training
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_inference_pipeline.py
│   └── test_reasoning_quality.py
│
├── configs/
│   ├── training_config.yaml            # batch size, lr, amount of data, etc.
│   └── model_config.yaml               # base model, reasoning token limit, etc.
│
├── README.md
└── requirements.txt
