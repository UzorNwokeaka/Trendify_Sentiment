import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():

    print("Loading vectorized data...")

    train_data = joblib.load("data/processed/tfidf_train.joblib")
    test_data = joblib.load("data/processed/tfidf_test.joblib")

    X_train = train_data["X_train"]
    y_train = train_data["y_train"]

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print("Training model...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # important for sentiment imbalance
        
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    # Save model
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, "artifacts/logistic_model.joblib")

    print("\n Model saved to artifacts/logistic_model.joblib")


if __name__ == "__main__":
    main()