# This script loads the dataset, cleans text, removes leakage terms,
# and saves the cleaned deliverable.

import os
import re
import argparse
import pandas as pd

from src.preprocessing import clean_text


def remove_leakage_terms(text: str) -> str:
    """
    Remove explicit rating-related leakage terms that can let the model
    'cheat' instead of learning true sentiment.

    Examples removed:
    - five stars
    - three stars
    - 5 stars
    - star / stars
    - sterne
    - étoile / étoiles
    """

    if not isinstance(text, str):
        return ""

    # Remove common rating phrases in English
    text = re.sub(
        r"\b(one|two|three|four|five|1|2|3|4|5)\s+stars?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove common multilingual star words / rating tokens
    text = re.sub(
        r"\b(star|stars|stern|sterne|etoile|etoiles|étoile|étoiles)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove spaced repeated rating words left behind after demojize / cleanup
    # e.g. "star star star"
    text = re.sub(
        r"\b(?:star|stars|stern|sterne|etoile|etoiles|étoile|étoiles)(?:\s+\b(?:star|stars|stern|sterne|etoile|etoiles|étoile|étoiles)\b)+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace after removals
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def main(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    required_cols = {
        "review_id",
        "product_category",
        "timestamp",
        "country",
        "rating",
        "review",
        "sentiment",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean raw review text
    df["review_clean"] = df["review"].apply(
        lambda x: clean_text(
            x,
            lowercase=True,
            fix_encoding=True,
            emoji_mode="demojize",
            keep_punctuation=True,
        )
    )

    # Remove leakage terms AFTER text cleaning
    df["review_clean"] = df["review_clean"].apply(remove_leakage_terms)

    # Drop rows that became empty after cleaning / leakage removal
    df = df[df["review_clean"].str.len() > 0].copy()

    # Remove exact duplicate cleaned reviews to reduce train/test memorization
    df = df.drop_duplicates(subset=["review_clean"]).copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    elif output_path.lower().endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        raise ValueError("Output must end with .parquet or .csv")

    print(f"✅ Cleaned dataset saved to: {output_path}")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print("\nSample cleaned reviews:")
    print(df[["review", "review_clean"]].head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1: Clean review text dataset")
    parser.add_argument(
        "--input",
        type=str,
        default=r"data\raw\reviews.csv",
        help="Path to raw input CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"data\processed\reviews_cleaned.parquet",
        help="Output path (.parquet or .csv)",
    )
    args = parser.parse_args()

    main(args.input, args.output)
    