import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score

def load_model_and_scaler():
    if not os.path.exists("spam_classifier_model.pkl") or not os.path.exists("spam_scaler.pkl"):
        raise FileNotFoundError("Trained model or scaler not found. Please train the model first.")
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

    output_df = pd.DataFrame({'Prediction': preds, 'Actual': y})
    output_df.to_csv("spam_predictions.csv", index=False)
    print(" Predictions saved to 'spam_predictions.csv'.")

def single_predict():
    model, scaler = load_model_and_scaler()
    print("\nEnter 57 comma-separated feature values:")
    user_input = input(" ").strip()

    try:
        values = list(map(float, user_input.split(',')))
        if len(values) != 57:
            raise ValueError("You must enter exactly 57 values.")
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        print(f"\n Prediction: {'SPAM' if pred == 1 else 'NOT SPAM'}")
    except Exception as e:
        print(f"\n Invalid input: {e}")
        print("Please enter exactly 57 comma-separated numeric values.")

def main():
    print("=== Spam Classifier CLI ===")
    print("Choose Mode:\n[1] Single Email Prediction\n[2] Batch Dataset Evaluation")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        single_predict()
    elif choice == '2':
        batch_predict()
    else:
        print(" Invalid option. Please choose 1 or 2.")

if __name__ == "__main__":
    main()
