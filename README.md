# Machine Learning Assignment 2

## a. Problem statement
Build and deploy an end-to-end classification workflow that compares the five classifiers explicitly listed in the assignment: Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random Forest. The application must allow a user to upload test data, select a model, and inspect evaluation results.

> Note: the assignment text says "all 6 ML models" in one sentence, but it enumerates five models and the marking/comparison tables also show five. This implementation follows the five explicitly named models.

## b. Dataset description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository (also bundled in scikit-learn as `load_breast_cancer`)  
**Problem type:** Binary classification  
**Instances:** 569  
**Features:** 30 numeric features  
**Classes:** malignant (0), benign (1)  
**Split:** 85% train / 15% test, stratified, random_state=42

This dataset satisfies the assignment minimum of 500 instances and 12 features.

## c. GitHub Repository Link
**Replace before submission:** `https://github.com/<your-username>/<your-repository>`

## d. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9767 | 0.9942 | 0.9815 | 0.9815 | 0.9815 | 0.9502 |
| kNN | 0.9535 | 0.9899 | 0.9630 | 0.9630 | 0.9630 | 0.9005 |
| Naive Bayes | 0.9419 | 0.9867 | 0.9623 | 0.9444 | 0.9533 | 0.8766 |
| Random Forest | 0.9419 | 0.9925 | 0.9623 | 0.9444 | 0.9533 | 0.8766 |
| Decision Tree | 0.9302 | 0.9062 | 0.9615 | 0.9259 | 0.9434 | 0.8536 |

### Performance observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performance in this run. Scaling plus a linear probabilistic boundary generalized very well on the 15% hold-out set. |
| Decision Tree | Lowest AUC/MCC among the five models in this run, suggesting a single tree was less stable than the other approaches. |
| kNN | Strong performance after standardization; local-neighbour structure worked well, although inference cost grows with the training set. |
| Naive Bayes | Fast baseline with good recall and F1, but the conditional-independence assumption limits flexibility. |
| Random Forest | Strong nonlinear ensemble with high AUC; robust but less interpretable than Logistic Regression. |
| Overall Winner for this dataset | **Logistic Regression**, selected by highest MCC while also considering accuracy/AUC. |

## Streamlit application
**Replace before submission:** `https://<your-app-name>.streamlit.app`

The app provides:
- CSV test-data upload
- model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1, and MCC
- confusion matrix
- classification report
- all-model comparison table

## Repository structure
```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
├── metadata.json
└── model/
    ├── train_models.py
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## How to run locally / on BITS Virtual Lab
```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## Deployment
Push all files to GitHub, then deploy `app.py` using Streamlit Community Cloud.

## Academic-integrity note
Run the full workflow yourself on BITS Virtual Lab, retain your own GitHub commit history, customize the README/UI/observations in your own words, and include your own BITS Virtual Lab execution screenshot in the final PDF.
