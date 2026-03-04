# creating Tokenization & Lemmatization scripts
import os
import argparse
import pandas as pd
import spacy


def tokenize_and_lemmatize_spacy(
    texts,
    nlp,
    batch_size: int = 1000,
    n_process: int = 1,
    remove_stop: bool = True,
):
    """
    Fast tokenization + lemmatization using nlp.pipe over an iterable of texts.
    Returns list[str] where each element is a space-joined lemma string.
    """
    results = []

    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = []
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue
            if not tok.is_alpha:
                continue
            if remove_stop and tok.is_stop:
                continue
            lemma = tok.lemma_.strip()
            if lemma:
                tokens.append(lemma)
        results.append(" ".join(tokens))
    return results


def main(input_path: str, output_path: str, text_col: str, batch_size: int, n_process: int):
    df = pd.read_csv(input_path)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")

    # Load spaCy model; disable components we don't need to speed things up.
    # We only need the tokenizer + lemmatizer.
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

    # Further speed: if your spaCy version supports it, keep only required parts
    # (en_core_web_sm includes tagger/attribute_ruler/lemmatizer)
    print("Tokenizing and lemmatizing text (fast nlp.pipe)...")

    df["review_lemma"] = tokenize_and_lemmatize_spacy(
        df[text_col].fillna("").astype(str).tolist(),
        nlp=nlp,
        batch_size=batch_size,
        n_process=n_process,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Lemmatized dataset saved to: {output_path}")
    print(df[[text_col, "review_lemma"]].head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Tokenization & Lemmatization (fast)")
    parser.add_argument("--input", type=str, default=r"data\processed\reviews_cleaned.csv")
    parser.add_argument("--output", type=str, default=r"data\processed\reviews_lemmatized.csv")
    parser.add_argument("--text_col", type=str, default="review_clean")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--n_process", type=int, default=1, help="Use >1 for multiprocessing (try 2 or 4).")
    args = parser.parse_args()

    main(args.input, args.output, args.text_col, args.batch_size, args.n_process)