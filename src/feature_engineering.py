# feature_engineering.py
# Converting the processed text into numerical representations using TF-IDF or contextual embeddings like BERT.
# This script creates vectorization (TF-IDF) and dimensionality reduction (PCA) pipelines for the sentiment analysis task.
import os
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(
    analyzer: str = "word",
    ngram_range=(1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int | None = 200_000,
):
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        strip_accents=None,   # keep accents (multilingual)
        lowercase=False,      # already cleaned/lowercased in Task 1
        sublinear_tf=True,    # often helps for text classification
    )


def main(
    input_path: str,
    text_col: str,
    label_col: str,
    artifacts_dir: str,
    processed_dir: str,
    test_size: float,
    random_state: int,
    analyzer: str,
    ngram_min: int,
    ngram_max: int,
):
    df = pd.read_csv(input_path)

    required = {text_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    # Basic cleanup
    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.len() > 0].copy()

    # Stratified split (keeps class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    vectorizer = build_vectorizer(
        analyzer=analyzer,
        ngram_range=(ngram_min, ngram_max),
    )

    print("Fitting TF-IDF vectorizer on training text...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Save vectorizer
    vec_path = os.path.join(artifacts_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)

    # Save vectorized matrices + labels (joblib is efficient for sparse matrices)
    joblib.dump(
        {"X_train": X_train_vec, "y_train": y_train.to_numpy()},
        os.path.join(processed_dir, "tfidf_train.joblib"),
    )
    joblib.dump(
        {"X_test": X_test_vec, "y_test": y_test.to_numpy()},
        os.path.join(processed_dir, "tfidf_test.joblib"),
    )

    print(f"Vectorizer saved: {vec_path}")
    print(f"Vectorized train saved: {os.path.join(processed_dir, 'tfidf_train.joblib')}")
    print(f"Vectorized test saved: {os.path.join(processed_dir, 'tfidf_test.joblib')}")
    print(f"Train shape: {X_train_vec.shape} | Test shape: {X_test_vec.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: TF-IDF Vectorization")
    parser.add_argument("--input", type=str, default=r"data\processed\reviews_cleaned.csv")
    parser.add_argument("--text_col", type=str, default="review_clean")  # or review_lemma
    parser.add_argument("--label_col", type=str, default="sentiment")

    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--processed_dir", type=str, default=r"data\processed")

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--analyzer", type=str, choices=["word", "char_wb"], default="word")
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)

    args = parser.parse_args()

    main(
        input_path=args.input,
        text_col=args.text_col,
        label_col=args.label_col,
        artifacts_dir=args.artifacts_dir,
        processed_dir=args.processed_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        analyzer=args.analyzer,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )