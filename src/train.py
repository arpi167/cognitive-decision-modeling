import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
from preprocess import generate_synthetic_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "labels.pkl")


def train():
    # Generate 1200 synthetic training rows
    df, le = generate_synthetic_data(n=1200, noise=0.10)

    X = df[["Logical_score", "Emotional_score", "Balanced_score"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # RandomForest is far more accurate than a single DecisionTree
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc * 100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"✅ Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()