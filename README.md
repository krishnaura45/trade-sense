# Trade sense : Jane street market modeling
Building real-time market forecasting models using deep tabular architectures and boosted tree ensembles on production-derived trading data.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Evaluator - Custom R2-score](https://img.shields.io/badge/Metric-Custom%20R2-blueviolet?style=for-the-badge)
![Metric Score](https://img.shields.io/badge/Best%20Score-0.007617-2ECC71?style=for-the-badge)
![Rank](https://img.shields.io/badge/Rank-376%20of%203757-brightgreen?style=for-the-badge)
![Solo](https://img.shields.io/badge/Submission-Solo-orange?style=for-the-badge)

### **Duration**: Jan 13, 2025 - April 8, 2025

---

## 🧠 Objective

The objective of this project was to forecast real-time market behavior using anonymized tabular features derived from Jane Street's production trading systems. The task was to model the target `responder_6`, capturing the expected return from a trade. The competition emphasized challenges like **non-stationarity**, **fat tails**, and **market shifts**. Evaluation was based on a **sample-weighted zero-mean R-squared score** — higher values indicating better predictive power.

---

## 🧩 My Approach

You can explore the complete methodology in this notebook: 🔗 [Jane Street - DL + Blend Ensemble](https://github.com/krishnaura45/tbp-skin-detect/blob/main/js-rmdf-sol-1.ipynb)

Key steps followed:

- 🔍 Explored Top Public Notebooks:
  - TabM + FTTransformer
  - Ridge baseline
  - XGB+NN blends with tuned weights

- 🪀 Deep Tabular Modeling:
  - Used custom TabM model with numerical embeddings for dense tabular data.
  - Leveraged pytorch-lightning for modular and reproducible training.

- 📊 Tree-Based Models:
  - Trained LGBM, XGBoost, and CatBoost regressors on cleaned features.
  - Tuned with early_stopping and controlled regularization.

- 🔄 Feature Selection & Preprocessing:
  - Dropped unstable low-variance features.
  - Split training using train_test_split, with careful tracking of data leakage.

- 💰 Weighted Model Ensemble:
  - Blended predictions from TabM, Ridge, and boosted models using custom-weighted average.
  - Weights guided by leaderboard feedback and validation R².

- 🔢 Zero-Mean Adjustment:
  - Applied mean subtraction on predictions to satisfy zero-mean constraint required by scoring.

---

## 🏆 Results / Outcomes

- ✅ Public Leaderboard Scores:
  - Range: *0.007520* to *0.008352* across 9 submissions

- 🏁 Private Leaderboard Scores:
  - Best: ***0.007617*** on final hidden test set

- 🥇 Rank Achieved:
  - Placed `376th` out of **3643 participants** and **3757 teams** as a **solo competitor**

---

## 🔗 References

- 🏆 Kaggle Competition: [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)

---

## 🛠️ Tech Stack

- Language: Python 🐍

- Libraries:
  - `numpy`, `pandas`, `polars` for data processing
  - `pytorch`, `pytorch-lightning` for deep learning
  - `lightgbm`, `xgboost`, `catboost` for tree-based regressors
  - `sklearn` for preprocessing & metrics

- Tools:
  - Jupyter Notebook / Kaggle Notebooks
  - GPU acceleration via PyTorch
