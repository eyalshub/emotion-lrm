#src/data/preprocess_goemotions.py
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/emotional_data.csv"
OUTPUT_DIR = "data/processed/"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "goemotions_clean_train.jsonl")
VAL_PATH = os.path.join(OUTPUT_DIR, "goemotions_clean_val.jsonl")

# List of emotions from the GoEmotions dataset
EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text

def preprocess_goemotions():
    print("Loading raw GoEmotions dataset...")
    df = pd.read_csv(RAW_PATH)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Filter out empty texts
    df = df[df["text"].str.len() > 1]

    # Create a label column
    df["label"] = df[EMOTIONS].idxmax(axis=1)

    # Select only the important columns
    df = df[["text", "label"]]

    print("Splitting train/validation...")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save_jsonl(df, path):
        with open(path, "w", encoding="utf8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps({
                    "text": row["text"],
                    "label": row["label"]
                }, ensure_ascii=False) + "\n")

    print("Saving processed files...")
    save_jsonl(train_df, TRAIN_PATH)
    save_jsonl(val_df, VAL_PATH)

    print("Done! Clean dataset saved.")

if __name__ == "__main__":
    preprocess_goemotions()
