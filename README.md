# Comparative Machine Learning Classification & Model Evaluation

This project evaluates and compares three supervised machine learning algorithms—**Logistic Regression**, **Support Vector Classifier (SVC)**, and a **Tuned Decision Tree**—to solve a binary classification problem. It focuses on dataset preprocessing, probability threshold optimization via cutoff curves, regularization, and comprehensive model reporting.

---

## Key Results & Model Comparison

All three models achieved strong, consistent performance across test data following feature scaling (`MinMaxScaler`) and hyperparameter tuning:

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **72%** | **0.72** | **0.72** | **0.72** |
| **Support Vector Classifier (SVC)** | **72%** | **0.71** | **0.73** | **0.72** |
| **Tuned Decision Tree** | **72%** | **0.72** | **0.72** | **0.72** |

### Key Findings
1. **Model Convergence:** Regularization and hyperparameter tuning allowed the Decision Tree to reach parity with linear models (72% accuracy across all three).
2. **Threshold Optimization:** Evaluating accuracy, sensitivity, and specificity across probability cutoffs identified **0.45** as the optimal threshold for operating precision and recall balance.
3. **Scaling Impact:** Applying `MinMaxScaler` ensured stable model training across SVC and Logistic Regression.

---

## Project Workflow

1. **Preprocessing & Feature Scaling:** Applied min-max scaling (`MinMaxScaler`) across features for linear and support vector models.
2. **Model Implementation:**
   * Tuned Decision Tree (baseline tree model)
   * Support Vector Classifier (SVC)
   * Logistic Regression ($L_2$ regularized default)
3. **Cutoff Optimization:** Generated Sensitivity/Specificity vs. Cutoff plots to select an optimal probability threshold ($0.45$).
4. **Evaluation:** Generated comparative classification reports and evaluated ROC AUC scores across all models.

---

## Environment & Dependencies

* Python 3.8+
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`

---

## How to Run

1. Clone the repository:
   ```bash
   clone https://github.com/sujaydas2030/supervised-classification-model-evaluation.git
