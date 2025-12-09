import os
from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "commentary_dataset.csv"
INDEX_PATH = BASE_DIR / "commentary_index.joblib"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Commentary dataset not found at: {DATA_PATH}")

    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)

    # We'll build the text we embed from these fields
    # Combine situation_tag + features_summary + commentary_text
    combined_text = (
        df["situation_tag"].fillna("") + " | " +
        df["features_summary"].fillna("") + " | " +
        df["commentary_text"].fillna("")
    )

    # 2) Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        lowercase=True,
        stop_words="english"
    )

    # 3) Fit and transform
    matrix = vectorizer.fit_transform(combined_text)

    # 4) Save everything we need for later retrieval
    index_payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "df": df,  # we keep the original rows to fetch commentary_text later
    }

    joblib.dump(index_payload, INDEX_PATH)
    print(f"Saved commentary index to: {INDEX_PATH}")


if __name__ == "__main__":
    main()
