# Comparative Machine Learning Classification & Model Evaluation

This project evaluates and compares three supervised machine learning algorithms—**Logistic Regression**, **Support Vector Classifier (SVC)**, and a **Tuned Decision Tree**—to solve a binary classification problem. It focuses on dataset preprocessing, probability threshold optimization via cutoff curves, regularization, and comprehensive model reporting.

---

## Key Results & Model Comparison

After feature scaling and probability cutoff tuning (optimal threshold: **0.45**), the model performances on test data were evaluated using accuracy, precision, recall, and F1-score:

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Performance Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **72%** | **0.72** | **0.72** | **0.72** | **Best Performing** |
| **Linear Support Vector Classifier** | **72%** | **0.71** | **0.73** | **0.72** | **Best Performing** |
| **Decision Tree (Tuned)** | 64% | 0.64 | 0.63 | 0.63 | Underperforming |

### Key Findings
1. **Linear Separation Supremacy:** Both Logistic Regression and Linear SVC outperformed the Tuned Decision Tree by **8 percentage points** in overall accuracy (72% vs. 64%).
2. **Threshold Optimization:** Evaluating accuracy, sensitivity, and specificity across probability cutoffs identified **0.45** as the optimal operating point to balance class precision and recall.
3. **Regularization:** Built-in $L_2$ regularization in Logistic Regression prevented overfitting effectively on the min-max scaled inputs.

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
