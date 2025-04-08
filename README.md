# 💳 Credit Card Fraud Detection

A machine learning project focused on identifying fraudulent credit card transactions using classification models and advanced data balancing techniques. Built with real-world imbalanced data, this project demonstrates proficiency in model evaluation, feature engineering, and business-aware data science practices.

---

## 🧠 Project Highlights

- Built binary classification models to predict fraudulent transactions in a highly imbalanced dataset.
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance.
- Compared **Logistic Regression** and **Random Forest** using evaluation metrics relevant to fraud detection (Recall, Precision, F1-score).
- Performed exploratory data analysis and preprocessing on anonymized transaction data.
- Achieved a recall of **84.5%** using Random Forest — critical for minimizing false negatives in fraud detection scenarios.

---

## 📂 Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraud Cases**: 492 (~0.17% of the dataset)
- **Features**: 30 anonymized columns (PCA-transformed), plus `Time`, `Amount`, and `Class`

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 99.2%    | 87.1%     | 76.3%  | 81.3%    |
| Random Forest       | **99.6%**| **91.7%** | **84.5%**| **88.0%** |

> 🎯 Focused on **Recall** and **Precision** to reflect the business need of minimizing false negatives (missed fraud cases).

---

## 🧰 Tools & Technologies

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn
- **Techniques**: Data preprocessing, Feature scaling, SMOTE, Confusion Matrix, ROC/AUC
- **Environment**: Jupyter Notebook

---

## 🚀 How to Run

```bash
git clone https://github.com/Aaditya0212/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt 
jupyter notebook project.ipynb