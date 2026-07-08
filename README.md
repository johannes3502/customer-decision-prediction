# Customer Decision Prediction

Comparative analysis of seven classifiers for predicting customer decisions across three real-world tabular datasets — from linear baselines to tree ensembles and neural networks.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project evaluates seven classification algorithms on three public datasets to predict binary customer outcomes (churn, term deposit subscription, and purchase intent). All models are trained with leak-free preprocessing and evaluated using stratified cross-validation or time-series splits.

**Algorithms**: Logistic Regression, k-Nearest Neighbors, Naive Bayes, SVM, Multi-Layer Perceptron, Random Forest, Gradient Boosting

**Key result**: Random Forest achieves **ROC-AUC 0.943** on the Bank Marketing dataset (nested cross-validation).

## Datasets

| Dataset | Source | Rows | Features | Target | Minority Class |
|---------|--------|------|----------|--------|----------------|
| Telco Customer Churn | IBM Community | 7,043 | 20 | Churn (y/n) | 26.5% |
| Bank Marketing | UCI / Moro et al. (2014) | 41,176 | 20 | Term deposit (y/n) | 11.3% |
| Online Shoppers Intention | UCI | 12,205 | 17 | Purchase (y/n) | 15.6% |

All three datasets exhibit class imbalance — **F1** and **ROC-AUC** are the primary evaluation metrics.

## Project Structure

```
├── configs/                         # Centralized YAML configuration
│   └── datasets.yaml                # Dataset paths, targets, split modes
├── data/                            # Generated locally (.gitignored)
│   ├── raw/                         # Downloaded CSVs
│   └── processed/                   # Cleaned CSVs (7-step pipeline)
├── notebooks/                       # Jupyter notebooks for exploration
│   ├── baseline_comparison.ipynb    # Baseline model comparison
│   ├── logreg_analysis.ipynb        # Logistic Regression
│   ├── knn_analysis.ipynb           # k-Nearest Neighbors
│   ├── naive_bayes_analysis.ipynb   # Naive Bayes
│   ├── svm_analysis.ipynb           # Support Vector Machine
│   ├── mlp_analysis.ipynb           # Multi-Layer Perceptron
│   ├── rf_analysis.ipynb            # Random Forest
│   ├── gradient_boosting_analysis.ipynb
│   ├── model_comparison.ipynb       # Cross-model comparison
│   └── model_analysis.ipynb         # Best model in-depth analysis
├── paper/
│   └── paper.pdf                    # Compiled project paper
├── reports/                         # Generated outputs
│   ├── plots/                       # All generated figures
│   └── results/                     # Result CSV tables
├── src/
│   ├── setup_data.py                # Data download + preprocessing
│   ├── analysis.py                  # Exploratory data analysis
│   ├── evaluation.py                # Baseline evaluation (LR, k-NN, NB)
│   ├── gradient_boosting.py         # Gradient Boosting evaluation
│   ├── model_analysis.py            # Analysis framework (16 functions)
│   ├── run_analysis.py              # Standalone analysis runner
│   ├── inspect_time_structure.py    # Temporal structure inspection
│   └── utils/
│       ├── download.py              # Dataset download utilities
│       └── preprocess.py            # 7-step preprocessing pipeline
├── tests/                           # Unit tests
│   ├── test_preprocess.py
│   └── test_download.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python src/setup_data.py    # Download + preprocess all datasets
```

## Usage

```bash
# Exploratory data analysis
python src/analysis.py

# Baseline classifier comparison (Logistic Regression, k-NN, Naive Bayes)
python src/evaluation.py

# Gradient Boosting evaluation (nested cross-validation)
python src/gradient_boosting.py

# Best model in-depth analysis
python src/run_analysis.py

# Interactive notebooks
jupyter notebook notebooks/
```

## Methodology

- **Leak-free preprocessing**: All scaling and encoding inside scikit-learn `Pipeline` objects, fitted exclusively on training folds
- **Nested cross-validation**: Inner loop for hyperparameter tuning (GridSearchCV), outer loop for unbiased generalization estimates
- **StratifiedKFold** (5-fold) preserves class distribution; **TimeSeriesSplit** (5-fold) for the temporally ordered Bank dataset
- **F1 and ROC-AUC** as primary metrics for imbalanced classification
- Random state fixed at `42` for reproducibility

### Preprocessing Pipeline

1. Remove exact duplicate rows
2. Normalize column names (spaces/slashes → underscores)
3. Strip leading/trailing whitespace from string cells
4. Replace placeholders (`unknown`, `?`, `999`) with NaN
5. Coerce numeric-looking object columns
6. Impute missing values: median (numeric), mode (categorical)
7. Encode target variable to binary `0`/`1`

## Results Summary

Baseline cross-validation results (5-fold, mean ± std):

| Dataset | Model | F1 | ROC-AUC |
|---------|-------|-----|---------|
| Telco | Logistic Regression | 0.592 ± 0.030 | 0.846 ± 0.013 |
| Telco | k-NN (k=5) | 0.547 ± 0.016 | 0.783 ± 0.007 |
| Telco | Naive Bayes | 0.597 ± 0.010 | 0.821 ± 0.012 |
| Bank | Logistic Regression | 0.475 ± 0.028 | 0.928 ± 0.018 |
| Bank | k-NN (k=5) | 0.417 ± 0.057 | 0.838 ± 0.032 |
| Bank | Naive Bayes | 0.350 ± 0.079 | 0.819 ± 0.047 |
| Ecom | Logistic Regression | 0.497 ± 0.027 | 0.890 ± 0.014 |
| Ecom | k-NN (k=5) | 0.496 ± 0.030 | 0.786 ± 0.011 |
| Ecom | Naive Bayes | 0.484 ± 0.018 | 0.812 ± 0.009 |

Advanced models (nested cross-validation ROC-AUC):

| Model | Telco | Bank | Ecom |
|-------|-------|------|------|
| MLP | 0.844 | 0.941 | 0.916 |
| Random Forest | 0.838 | **0.943** | 0.924 |
| Gradient Boosting | 0.848 | 0.945 | 0.932 |

**Best model**: Random Forest on Bank Marketing (ROC-AUC 0.943, nested CV, parameters: `n_estimators=200`, `max_depth=20`, `max_features='sqrt'`, `class_weight='balanced'`).

## Model Analysis

The `model_analysis.py` framework provides 16 model-agnostic functions for in-depth analysis:

| Analysis | Description |
|----------|-------------|
| Permutation Importance | Feature importance via F1 decrease on shuffling |
| SHAP | TreeExplainer for local and global feature attribution |
| Learning Curves | Train/validation score vs. training set size |
| Confusion Matrix | Absolute and normalized class-level metrics |
| Hyperparameter Sensitivity | Cross-validation score vs. parameter values |
| Misclassification Patterns | Feature mean differences: correct vs. incorrect predictions |

## License

MIT — see [LICENSE](LICENSE) for details.

Dataset sources: Telco (IBM Community License), Bank (CC BY 4.0), Ecom (UCI, public domain).
