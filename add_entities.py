# add_entities.py

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def extract_entities(texts: list[str]) -> list[str]:
    ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    entities_col = []

    for text in tqdm(texts, desc="Extracting Entities"):
        try:
            result = ner(text)
            # Filter only people, orgs, locations, etc.
            entities = [ent["word"] for ent in result if ent["entity_group"] in ["PER", "ORG", "LOC", "MISC"]]
            entities = list(set(entities))  # remove duplicates
            entities_col.append(", ".join(entities))
        except Exception as e:
            print(f"⚠️ Error processing text: {e}")
            entities_col.append("")

    return entities_col

def save_augmented_dataset(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    print(f"✅ Saved augmented dataset to: {output_path}")

def main():
    INPUT_CSV = "neon_labeled.csv"
    OUTPUT_CSV = "neon_labeled_with_entities.csv"

    print(f"📥 Loading dataset from: {INPUT_CSV}")
    df = load_dataset(INPUT_CSV)

    if "text" not in df.columns:
        print("❌ Column 'text' not found in the CSV.")
        return

    df["target_entities"] = extract_entities(df["text"].tolist())
    save_augmented_dataset(df, OUTPUT_CSV)

if __name__ == "__main__":
    main()
