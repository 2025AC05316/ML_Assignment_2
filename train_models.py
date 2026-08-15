"""
BITS Pilani - Machine Learning Assignment 2
Training script for five required classifiers using the
Breast Cancer Wisconsin (Diagnostic) dataset.

Run on BITS Virtual Lab:
    python model/train_models.py
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
TEST_SIZE = 0.15

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=RANDOM_STATE
    ),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    ),
}

rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })

    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, MODEL_DIR / f"{filename}.joblib")

metrics = pd.DataFrame(rows)
metrics.to_csv(ROOT / "model_metrics.csv", index=False)

test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(ROOT / "test_data.csv", index=False)

metadata = {
    "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
    "dataset_source": "UCI Machine Learning Repository",
    "n_instances": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "class_names": list(map(str, data.target_names)),
    "target_column": "target",
    "feature_names": list(X.columns),
    "split": "85/15",
    "random_state": RANDOM_STATE,
}
(ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2))

print(metrics.round(4).to_string(index=False))
print("\nSaved models, test_data.csv, model_metrics.csv, and metadata.json.")
