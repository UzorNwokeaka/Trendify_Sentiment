import os
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    print("Loading vectorized data...")

    train_data = joblib.load("data/processed/tfidf_train.joblib")
    test_data = joblib.load("data/processed/tfidf_test.joblib")

    X_train = train_data["X_train"]
    y_train = train_data["y_train"]

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print("Training Naive Bayes model...")

    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("Evaluating model...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:\n")
    print(cm)

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(model, "artifacts/naive_bayes_model.joblib")

    with open("reports/naive_bayes_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("\n Model saved to artifacts/naive_bayes_model.joblib")
    print("Report saved to reports/naive_bayes_classification_report.txt")


if __name__ == "__main__":
    main()