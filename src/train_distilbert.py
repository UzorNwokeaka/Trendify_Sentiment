import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


MODEL_NAME = "distilbert-base-multilingual-cased"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def prepare_dataset(csv_path: str, text_col: str = "review_clean", label_col: str = "sentiment"):
    df = pd.read_csv(csv_path)

    required_cols = {text_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[[text_col, label_col]].dropna().copy()
    df = df[df[text_col].str.len() > 0].copy()

    # Encode labels
    df["label"] = df[label_col].map(LABEL2ID)

    if df["label"].isna().any():
        bad = df[df["label"].isna()][label_col].unique()
        raise ValueError(f"Unexpected labels found: {bad}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_function(examples, tokenizer, text_col):
    return tokenizer(
        examples[text_col],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def main():
    input_path = "data/processed/reviews_cleaned.csv"
    output_dir = "artifacts/distilbert_model"
    report_dir = "reports"
    text_col = "review_clean"
    label_col = "sentiment"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print("Loading and preparing dataset...")
    train_df, test_df = prepare_dataset(input_path, text_col=text_col, label_col=label_col)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(train_df[[text_col, "label"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[[text_col, "label"]], preserve_index=False)

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer, text_col), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(x, tokenizer, text_col), batched=True)

    train_ds = train_ds.remove_columns([text_col])
    test_ds = test_ds.remove_columns([text_col])

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("Training DistilBERT...")
    trainer.train()

    print("Evaluating model...")
    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(3)],
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:\n")
    print(cm)

    print("Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(report_dir, "distilbert_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nModel saved to: {output_dir}")
    print(f"Report saved to: {os.path.join(report_dir, 'distilbert_classification_report.txt')}")


if __name__ == "__main__":
    main()