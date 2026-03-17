import sys
import re
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.preprocessing import clean_text


# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LR_MODEL_PATH = ARTIFACTS_DIR / "logistic_model.joblib"
NB_MODEL_PATH = ARTIFACTS_DIR / "naive_bayes_model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
DISTILBERT_MODEL_DIR = ARTIFACTS_DIR / "distilbert_model"

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {
    "positive": "#16a34a",
    "neutral": "#6b7280",
    "negative": "#dc2626",
}


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "review" in df.columns:
        df["review"] = df["review"].astype(str)
        df["review_length"] = df["review"].str.len()
        df["word_count"] = df["review"].str.split().str.len()

    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.lower()

    return df


# -----------------------------
# Artifact loading
# -----------------------------
@st.cache_resource
def load_sklearn_artifacts():
    artifacts = {
        "vectorizer": None,
        "models": {},
    }

    if VECTORIZER_PATH.exists():
        artifacts["vectorizer"] = joblib.load(VECTORIZER_PATH)

    if LR_MODEL_PATH.exists():
        artifacts["models"]["Logistic Regression"] = joblib.load(LR_MODEL_PATH)

    if NB_MODEL_PATH.exists():
        artifacts["models"]["Naive Bayes"] = joblib.load(NB_MODEL_PATH)

    return artifacts


@st.cache_resource
def load_distilbert_artifacts():
    if not DISTILBERT_MODEL_DIR.exists():
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_DIR)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }


# -----------------------------
# Training-time leakage removal
# -----------------------------
def remove_leakage_terms(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"\b(one|two|three|four|five|1|2|3|4|5)\s+stars?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(star|stars|stern|sterne|etoile|etoiles|étoile|étoiles)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def preprocess_for_sklearn(text: str) -> str:
    cleaned = clean_text(
        text,
        lowercase=True,
        fix_encoding=True,
        emoji_mode="demojize",
        keep_punctuation=True,
    )
    cleaned = remove_leakage_terms(cleaned)
    return cleaned.strip()


def preprocess_for_transformer(text: str) -> str:
    cleaned = clean_text(
        text,
        lowercase=True,
        fix_encoding=True,
        emoji_mode="demojize",
        keep_punctuation=True,
    )
    cleaned = remove_leakage_terms(cleaned)
    return cleaned.strip()


# -----------------------------
# Helper functions
# -----------------------------
def reorder_sentiments(labels):
    labels = [str(x).lower() for x in labels]
    ordered = [s for s in SENTIMENT_ORDER if s in labels]
    remaining = [s for s in labels if s not in ordered]
    return ordered + remaining


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def sentiment_distribution_fig(df: pd.DataFrame):
    counts = df["sentiment"].value_counts()
    order = reorder_sentiments(counts.index.tolist())
    counts = counts.reindex(order)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[SENTIMENT_COLORS.get(x, "#2563eb") for x in counts.index],
            text=counts.values,
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Number of Reviews",
        height=400,
    )
    return fig


def sentiment_by_country_fig(df: pd.DataFrame):
    ctab = pd.crosstab(df["country"], df["sentiment"], normalize="index") * 100
    if "positive" in ctab.columns:
        ctab = ctab.sort_values("positive", ascending=False)

    fig = go.Figure()
    for sentiment in SENTIMENT_ORDER:
        if sentiment in ctab.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment.capitalize(),
                    x=ctab.index,
                    y=ctab[sentiment].values,
                    marker_color=SENTIMENT_COLORS[sentiment],
                )
            )

    fig.update_layout(
        title="Sentiment Distribution by Country",
        xaxis_title="Country",
        yaxis_title="Percentage",
        barmode="stack",
        height=450,
    )
    return fig


def sentiment_by_category_fig(df: pd.DataFrame):
    ctab = pd.crosstab(df["product_category"], df["sentiment"], normalize="index") * 100
    if "positive" in ctab.columns:
        ctab = ctab.sort_values("positive", ascending=False)

    fig = go.Figure()
    for sentiment in SENTIMENT_ORDER:
        if sentiment in ctab.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment.capitalize(),
                    y=ctab.index,
                    x=ctab[sentiment].values,
                    orientation="h",
                    marker_color=SENTIMENT_COLORS[sentiment],
                )
            )

    fig.update_layout(
        title="Sentiment Distribution by Product Category",
        xaxis_title="Percentage",
        yaxis_title="Category",
        barmode="stack",
        height=600,
    )
    return fig


def rating_distribution_fig(df: pd.DataFrame):
    counts = df["rating"].value_counts().sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            text=counts.values,
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Rating Distribution",
        xaxis_title="Rating",
        yaxis_title="Count",
        height=400,
    )
    return fig


# -----------------------------
# Prediction functions
# -----------------------------
def predict_sklearn(text: str, model, vectorizer):
    processed = preprocess_for_sklearn(text)

    if not processed.strip():
        processed = "empty_review"

    X = vectorizer.transform([processed])

    pred = model.predict(X)[0]
    pred = str(pred).lower()

    prob_df = None

    # First try real probabilities
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)[0]
            classes = [str(c).lower() for c in model.classes_]

            prob_df = pd.DataFrame(
                {
                    "sentiment": classes,
                    "probability": probs,
                }
            )
        except Exception:
            prob_df = None

    # Fallback: derive pseudo-probabilities from decision_function
    if prob_df is None and hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)

            if scores.ndim == 1:
                # binary case fallback
                probs_pos = 1 / (1 + np.exp(-scores[0]))
                probs = np.array([1 - probs_pos, probs_pos])
                classes = [str(c).lower() for c in model.classes_]
            else:
                probs = softmax(scores[0])
                classes = [str(c).lower() for c in model.classes_]

            prob_df = pd.DataFrame(
                {
                    "sentiment": classes,
                    "probability": probs,
                }
            )
        except Exception:
            prob_df = None

    if prob_df is not None:
        prob_df["sort_order"] = prob_df["sentiment"].apply(
            lambda x: SENTIMENT_ORDER.index(x) if x in SENTIMENT_ORDER else 999
        )
        prob_df = prob_df.sort_values("sort_order").drop(columns="sort_order").reset_index(drop=True)

    return pred, prob_df, processed


def predict_distilbert(text: str, model, tokenizer, device: str):
    processed = preprocess_for_transformer(text)

    if not processed.strip():
        processed = "empty review"

    inputs = tokenizer(
        processed,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    # DistilBERT does not use token_type_ids
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().tolist()[0]

    pred_idx = int(np.argmax(probs))

    id2label = model.config.id2label
    labels = [str(id2label[i]).lower() for i in range(len(probs))]
    pred = labels[pred_idx]

    prob_df = pd.DataFrame(
        {
            "sentiment": labels,
            "probability": probs,
        }
    )
    prob_df["sort_order"] = prob_df["sentiment"].apply(
        lambda x: SENTIMENT_ORDER.index(x) if x in SENTIMENT_ORDER else 999
    )
    prob_df = prob_df.sort_values("sort_order").drop(columns="sort_order").reset_index(drop=True)

    return pred, prob_df, processed


# -----------------------------
# Pages
# -----------------------------
def page_dashboard(df: pd.DataFrame):
    st.title("Trendify Global — Sentiment Dashboard")
    st.caption("Exploratory analysis of customer feedback")

    with st.sidebar:
        st.subheader("Filters")

        categories = ["All"] + sorted(df["product_category"].dropna().unique().tolist())
        countries = ["All"] + sorted(df["country"].dropna().unique().tolist())

        selected_category = st.selectbox("Product Category", categories)
        selected_country = st.selectbox("Country", countries)

    dff = df.copy()

    if selected_category != "All":
        dff = dff[dff["product_category"] == selected_category]

    if selected_country != "All":
        dff = dff[dff["country"] == selected_country]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(dff):,}")
    c2.metric("Positive", f"{(dff['sentiment'] == 'positive').sum():,}")
    c3.metric("Neutral", f"{(dff['sentiment'] == 'neutral').sum():,}")
    c4.metric("Negative", f"{(dff['sentiment'] == 'negative').sum():,}")

    st.plotly_chart(sentiment_distribution_fig(dff), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(rating_distribution_fig(dff), use_container_width=True)
    with col2:
        st.plotly_chart(sentiment_by_country_fig(dff), use_container_width=True)

    st.plotly_chart(sentiment_by_category_fig(dff), use_container_width=True)

    with st.expander("Preview data"):
        st.dataframe(dff.head(50), use_container_width=True)


def page_predict():
    st.title("Sentiment Prediction")
    st.caption("Enter a review and choose a model to predict sentiment.")

    sklearn_artifacts = load_sklearn_artifacts()
    distilbert_artifacts = load_distilbert_artifacts()

    model_options = list(sklearn_artifacts["models"].keys())
    if distilbert_artifacts is not None:
        model_options.append("DistilBERT")

    if not model_options:
        st.error("No models found in artifacts/.")
        return

    selected_model = st.selectbox("Choose model", model_options)

    user_text = st.text_area(
        "Customer review",
        height=160,
        placeholder="Type a customer review here...",
    )

    show_processed = st.checkbox("Show processed text", value=True)
    show_debug = st.checkbox("Show debug info", value=False)

    if st.button("Predict", type="primary"):
        if not user_text.strip():
            st.warning("Please enter review text.")
            return

        try:
            if selected_model == "DistilBERT":
                pred, prob_df, processed = predict_distilbert(
                    user_text,
                    distilbert_artifacts["model"],
                    distilbert_artifacts["tokenizer"],
                    distilbert_artifacts["device"],
                )
            else:
                vectorizer = sklearn_artifacts["vectorizer"]
                model = sklearn_artifacts["models"][selected_model]

                if vectorizer is None:
                    st.error("TF-IDF vectorizer not found.")
                    return

                pred, prob_df, processed = predict_sklearn(user_text, model, vectorizer)

            if pred == "positive":
                st.success(f"Prediction: {pred.upper()}")
            elif pred == "negative":
                st.error(f"Prediction: {pred.upper()}")
            else:
                st.info(f"Prediction: {pred.upper()}")

            if show_processed:
                st.subheader("Processed Text")
                st.code(processed, language="text")

            if prob_df is not None:
                st.subheader("Prediction Confidence")

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=prob_df["sentiment"],
                        y=prob_df["probability"],
                        marker_color=[SENTIMENT_COLORS.get(x, "#2563eb") for x in prob_df["sentiment"]],
                        text=(prob_df["probability"] * 100).round(1).astype(str) + "%",
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    xaxis_title="Sentiment",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1]),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(prob_df, use_container_width=True)

            if show_debug:
                st.write("Selected model:", selected_model)
                st.write("Processed text:", processed)
                st.write("Predicted label:", pred)
                if prob_df is not None:
                    st.dataframe(prob_df, use_container_width=True)

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(
        page_title="Trendify Global Sentiment App",
        page_icon="🛒",
        layout="wide",
    )

    st.sidebar.title("Trendify Global")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Predict"])

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    df = load_data(DATA_PATH)

    required_cols = {"review", "sentiment", "rating", "country", "product_category"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in dataset: {missing}")
        st.stop()

    if page == "Dashboard":
        page_dashboard(df)
    else:
        page_predict()


if __name__ == "__main__":
    main()