# Click-to-Cart: A Predictive Modeling Analysis

This repository contains the code and analysis for a binary classification task to predict whether an e-commerce customer will make a purchase. The project follows a complete data science workflow, from exploratory data analysis (EDA) to feature engineering, model comparison, and visualization.

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/bholu777/click-to-cart-prediction/blob/main/Model.ipynb)

## 1. Project Overview

The goal was to build a model that predicts if a customer will purchase (`1`) or not (`0`) based on their browsing behavior.

* **Dataset:** 2,250 rows of e-shopping data.
* **Features:** `Time_on_site`, `Pages_viewed`, `Clicked_ad`, `Cart_value`, `Referral` (categorical), `Last_Ad_Seen` (categorical).
* **Target:** `Purchase` (1 or 0).
* **Problem:** Binary Classification.

### Dataset
The dataset used for this analysis can be found here:
* **Dataset**: [Training_and_Test](https://drive.google.com/drive/u/0/folders/1bDGWF8N644N-aG0zAoK6D7PuxMD1SQRK)

## 2. Methodology & Analytical Choices

This project's focus was not just on building a model, but on making intelligent, justified decisions during the data preprocessing pipeline.

### Exploratory Data Analysis (EDA)
* Performed a full analysis of the data's mathematical features, null values, and distributions.
* Used `seaborn` and `matplotlib` to visualize numerical (box plots) and categorical (count plots) features against the `Purchase` target.
* Analyzed the target variable and found a 70:30 class split, which is reasonably balanced. No over/under-sampling (like SMOTE) was necessary.

### Feature Engineering
1.  **Feature Selection:** Dropped `Browser_Refresh_Rate` due to a near-zero correlation (-0.005) with the target.
2.  **Categorical Encoding:** Applied `OneHotEncoder` (via `ColumnTransformer`) to the `Referral` and `Last_Ad_Seen` columns.
3.  **Outlier Handling:** Outliers in `Cart_value` and `Pages_viewed` were intentionally kept. They represent realistic user behavior (e.g., adding many items but not buying) and are not just data errors.
4.  **Data Scaling:** Standardization/Normalization was not performed. The chosen models (Random Forest, XGBoost) are tree-based and not distance-sensitive, so scaling is computationally unnecessary.

## 3. Model Comparison: Random Forest vs. XGBoost

Two powerful ensemble models were trained and compared.

### Performance Results

While both models achieved an identical **Accuracy of 79.11%**, a deeper look at other metrics was required to select the superior model.

| Metric | Random Forest | XGBoost |
| :--- | :---: | :---: |
| **Accuracy** | 79.11% | 79.11% |
| **F1 Score** | 0.61 | 0.58 |
| **AUC Score** | 0.845 | 0.854 |
| True Positive Rate | 0.5441 | 0.4853 |
| False Positive Rate | 0.0764 | 0.1019 |

### Analysis
* Random Forest showed a better F1 Score and a higher True Positive Rate (Recall), meaning it was better at correctly identifying customers who would make a purchase.
* XGBoost achieved a higher AUC Score (0.854), indicating a slightly better overall ability to distinguish between the two classes across all thresholds.

| Random Forest (AUC: 0.845) | XGBoost (AUC: 0.854) |
| :---: | :---: |

## 4. Conclusion & Visualization

Based on its superior AUC score, XGBoost was selected as the preferred model for this task, as it demonstrates a more robust classification ability.

To visualize the model's complex, non-linear decision boundary, the 10-dimensional data was projected into 2 dimensions using Principal Component Analysis (PCA).

### Hyperparameters
The final models were tuned using `RandomizedSearchCV`. Key parameters included:
* **Random Forest:** `n_estimators=200`, `max_depth=8`, `criterion='log_loss'`
* **XGBoost:** `max_depth=3`, `eta=0.1`, `n_estimators=100`, `subsample=0.8`
