import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="ML Classification Model Evaluation",
    page_icon="🧠",
    layout="wide"
)

RANDOM_STATE = 42


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

@st.cache_data
def load_data():

    data = load_breast_cancer(as_frame=True)

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        list(X.columns),
        list(data.target_names)
    )


# ---------------------------------------------------------
# Train Models
# ---------------------------------------------------------

@st.cache_resource
def train_models(X_train, y_train):

    models = {

        "Logistic Regression":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model",
                 LogisticRegression(
                     max_iter=5000,
                     random_state=RANDOM_STATE
                 ))
            ]),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=5,
                random_state=RANDOM_STATE
            ),

        "kNN":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model",
                 KNeighborsClassifier(
                     n_neighbors=7
                 ))
            ]),

        "Naive Bayes":
            GaussianNB(),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    class_names
) = load_data()


# ---------------------------------------------------------
# Train all five models
# ---------------------------------------------------------

models = train_models(
    X_train,
    y_train
)


# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------

st.title(
    "🧠 ML Classification Model Evaluation Lab"
)

st.write(
    "Breast Cancer Wisconsin (Diagnostic) Dataset"
)

st.success(
    "All five ML models have been trained successfully."
)


# ---------------------------------------------------------
# Model Selection
# ---------------------------------------------------------

selected_model_name = st.selectbox(
    "Select Classification Model",
    list(models.keys())
)

model = models[
    selected_model_name
]


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ---------------------------------------------------------
# Calculate six required metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_probability
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ---------------------------------------------------------
# Display metrics
# ---------------------------------------------------------

st.subheader(
    f"{selected_model_name} Performance"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

c2.metric(
    "AUC",
    f"{auc:.4f}"
)

c3.metric(
    "Precision",
    f"{precision:.4f}"
)

c4.metric(
    "Recall",
    f"{recall:.4f}"
)

c5.metric(
    "F1",
    f"{f1:.4f}"
)

c6.metric(
    "MCC",
    f"{mcc:.4f}"
)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

st.subheader(
    "Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots()

image = ax.imshow(cm)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_xticks(
    [0, 1],
    class_names
)

ax.set_yticks(
    [0, 1],
    class_names
)

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

fig.colorbar(
    image,
    ax=ax
)

st.pyplot(fig)


# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

st.subheader(
    "Classification Report"
)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)


# ---------------------------------------------------------
# Test CSV Upload
# ---------------------------------------------------------

st.subheader(
    "Upload Test Data"
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(
        uploaded_file
    )

    st.write(
        "Uploaded Data"
    )

    st.dataframe(
        uploaded_df.head()
    )

    missing_columns = [
        column
        for column in feature_names
        if column not in uploaded_df.columns
    ]

    if missing_columns:

        st.error(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    else:

        uploaded_X = uploaded_df[
            feature_names
        ]

        predictions = model.predict(
            uploaded_X
        )

        probabilities = model.predict_proba(
            uploaded_X
        )[:, 1]

        result = uploaded_df.copy()

        result[
            "Prediction"
        ] = predictions

        result[
            "Prediction Label"
        ] = [
            class_names[int(x)]
            for x in predictions
        ]

        result[
            "Probability"
        ] = probabilities

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            result,
            use_container_width=True
        )


# ---------------------------------------------------------
# Compare all models
# ---------------------------------------------------------

st.subheader(
    "All Model Comparison"
)

comparison = []

for name, current_model in models.items():

    pred = current_model.predict(
        X_test
    )

    probability = current_model.predict_proba(
        X_test
    )[:, 1]

    comparison.append({

        "ML Model Name":
            name,

        "Accuracy":
            accuracy_score(
                y_test,
                pred
            ),

        "AUC":
            roc_auc_score(
                y_test,
                probability
            ),

        "Precision":
            precision_score(
                y_test,
                pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                pred,
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
                y_test,
                pred
            )
    })


comparison_df = pd.DataFrame(
    comparison
)

st.dataframe(
    comparison_df,
    use_container_width=True
)


# ---------------------------------------------------------
# Find best model
# ---------------------------------------------------------

best_index = comparison_df[
    "MCC"
].idxmax()

best_model = comparison_df.loc[
    best_index,
    "ML Model Name"
]

best_mcc = comparison_df.loc[
    best_index,
    "MCC"
]

st.success(
    f"Best model based on MCC: "
    f"{best_model} "
    f"(MCC = {best_mcc:.4f})"
)
