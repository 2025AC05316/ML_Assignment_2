from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Loan/Medical Risk Classification Model Lab",
    page_icon="🧠",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

metadata = json.loads((ROOT / "metadata.json").read_text())
reference_metrics = pd.read_csv(ROOT / "model_metrics.csv")

st.title("🧠 ML Classification Model Evaluation Lab")
st.caption(
    "BITS Pilani Machine Learning Assignment 2 | "
    "Breast Cancer Wisconsin (Diagnostic) dataset"
)

with st.sidebar:
    st.header("Controls")
    selected_model_name = st.selectbox(
        "Select classification model",
        list(MODEL_FILES.keys())
    )
    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help="Use the supplied test_data.csv or another CSV with the same feature columns. "
             "Include a 'target' column to calculate evaluation metrics."
    )

    st.markdown("---")
    st.markdown("**Dataset requirements met**")
    st.write(f"Instances: {metadata['n_instances']}")
    st.write(f"Features: {metadata['n_features']}")
    st.write(f"Split: {metadata['split']}")

tab1, tab2, tab3 = st.tabs(
    ["Model Evaluation", "All-Model Comparison", "Dataset Guide"]
)

with tab1:
    st.subheader(selected_model_name)

    if uploaded_file is None:
        st.info("Upload `test_data.csv` from the repository to evaluate the selected model.")
    else:
        df = pd.read_csv(uploaded_file)

        required_features = metadata["feature_names"]
        missing = [c for c in required_features if c not in df.columns]

        if missing:
            st.error(
                "Uploaded file is missing required feature columns:\n\n"
                + ", ".join(missing)
            )
        else:
            model = joblib.load(MODEL_FILES[selected_model_name])
            X = df[required_features]

            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            result_df = df.copy()
            result_df["prediction"] = y_pred
            result_df["prediction_label"] = [
                metadata["class_names"][int(v)] for v in y_pred
            ]
            result_df["positive_class_probability"] = y_prob

            st.markdown("#### Prediction preview")
            st.dataframe(result_df.head(20), use_container_width=True)

            if metadata["target_column"] in df.columns:
                y_true = df[metadata["target_column"]].astype(int)

                metrics = {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "AUC": roc_auc_score(y_true, y_prob),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall": recall_score(y_true, y_pred, zero_division=0),
                    "F1": f1_score(y_true, y_pred, zero_division=0),
                    "MCC": matthews_corrcoef(y_true, y_pred),
                }

                cols = st.columns(6)
                for col, (name, value) in zip(cols, metrics.items()):
                    col.metric(name, f"{value:.4f}")

                st.markdown("#### Confusion matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4.8, 4.0))
                im = ax.imshow(cm)
                ax.set_xticks([0, 1], metadata["class_names"])
                ax.set_yticks([0, 1], metadata["class_names"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, int(cm[i, j]), ha="center", va="center")
                fig.colorbar(im, ax=ax)
                st.pyplot(fig)

                st.markdown("#### Classification report")
                report = classification_report(
                    y_true, y_pred,
                    target_names=metadata["class_names"],
                    output_dict=True,
                    zero_division=0
                )
                st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            else:
                st.warning(
                    "No target column was found, so predictions are shown but evaluation "
                    "metrics cannot be calculated."
                )

with tab2:
    st.subheader("Reference performance on the fixed 15% hold-out test set")
    display = reference_metrics.copy()
    numeric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    display[numeric_cols] = display[numeric_cols].round(4)
    st.dataframe(display, use_container_width=True)

    best_idx = reference_metrics["MCC"].idxmax()
    best_name = reference_metrics.loc[best_idx, "ML Model Name"]
    best_mcc = reference_metrics.loc[best_idx, "MCC"]
    st.success(f"Overall winner by MCC: {best_name} (MCC = {best_mcc:.4f})")

with tab3:
    st.markdown(
        """
        **Expected CSV structure**
        - 30 numeric predictor columns matching the Breast Cancer Wisconsin (Diagnostic) features.
        - Optional `target` column:
          - `0` = malignant
          - `1` = benign

        **What this app demonstrates**
        - CSV upload for test data
        - model-selection dropdown
        - six required evaluation metrics
        - confusion matrix
        - classification report
        - comparison of all five models specified in the assignment
        """
    )
