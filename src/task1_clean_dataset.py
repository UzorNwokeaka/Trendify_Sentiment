# This script loads my dataset, cleans text, and saves the deliverable.

import os
import argparse
import pandas as pd

from src.preprocessing import clean_text


def main(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    required_cols = {"review_id", "product_category", "timestamp", "country", "rating", "review", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["review_clean"] = df["review"].apply(
        lambda x: clean_text(
            x,
            lowercase=True,
            fix_encoding=True,
            emoji_mode="demojize",
            keep_punctuation=True,
        )
    )

    df = df[df["review_clean"].str.len() > 0].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    elif output_path.lower().endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        raise ValueError("Output must end with .parquet or .csv")

    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print("\nSample cleaned reviews:")
    print(df[["review", "review_clean"]].head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1: Clean review text dataset")
    parser.add_argument("--input", type=str, default=r"data\raw\reviews.csv", help="Path to raw input CSV")
    parser.add_argument("--output", type=str, default=r"data\processed\reviews_cleaned.parquet", help="Output path (.parquet or .csv)")
    args = parser.parse_args()

    main(args.input, args.output)
