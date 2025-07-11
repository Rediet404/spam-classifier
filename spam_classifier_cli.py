import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score


def load_model_and_scaler():
    print("Loading model and scaler...")
    model = joblib.load("spam_classifier_model.pkl")
    scaler = joblib.load("spam_scaler.pkl")
    return model, scaler


def load_dataset():
    path = "data/spambase.data"
    if os.path.exists(path):
        df = pd.read_csv(path, header=None)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        df = pd.read_csv(url, header=None)

    df.columns = [f'feature_{i}' for i in range(57)] + ['label']
    return df


def batch_predict():
    model, scaler = load_model_and_scaler()
    df = load_dataset()
    X = df.drop('label', axis=1)
    y = df['label']

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    print("\n Evaluation on Full Dataset:")
    print(classification_report(y, preds))
    print(f" Accuracy: {accuracy_score(y, preds):.4f}")

    pd.DataFrame({'Prediction': preds, 'Actual': y}).to_csv("spam_predictions.csv", index=False)
    print(" Saved predictions to spam_predictions.csv")


def single_predict():
    model, scaler = load_model_and_scaler()

    print("\nEnter 57 comma-separated values for a single email:")
    raw_input = input("👉 ").strip()

    try:
        features = list(map(float, raw_input.split(',')))
        if len(features) != 57:
            raise ValueError("Expected exactly 57 values.")
        X = pd.DataFrame([features], columns=[f'feature_{i}' for i in range(57)])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        label = "SPAM" if pred == 1 else "NOT SPAM"
        print(f"\n Prediction: {label}")
    except Exception as e:
        print(f"\n Error: {e}")
        print("Make sure to enter exactly 57 numerical values, separated by commas.")


def main():
    print("=== Spam Classifier CLI ===")
    print("Choose mode: [1] Single Email  [2] Batch Dataset")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        single_predict()
    elif choice == "2":
        batch_predict()
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
