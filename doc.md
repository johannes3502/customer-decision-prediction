# Project Documentation — Customer Decision Prediction

Comparative analysis of seven classifiers across three real-world tabular datasets to predict
binary customer decisions. The project follows a four-phase pipeline: data preparation,
baseline evaluation, advanced modeling, and in-depth model analysis.

---

## 1. Datasets

| Dataset | Source | Rows | Features | Target | Minority |
|---------|--------|------|----------|--------|----------|
| Telco Customer Churn | Kaggle / IBM | 7,043 | 20 | Churn | 26.5% |
| Bank Marketing | UCI | 41,176 | 20 | Term deposit | 11.3% |
| Online Shoppers Intention | UCI | 12,205 | 17 | Purchase | 15.6% |

All datasets are imbalanced — accuracy alone is misleading. F1 and ROC-AUC are the
primary evaluation metrics.

---

## 2. Pipeline Architecture

### 2.1 Data Pipeline

```
Download (download.py) → Preprocessing (setup_data.py, 7 steps) → Cleaned CSVs
```

7-step preprocessing:
1. Remove exact duplicate rows
2. Normalize column names (strip, spaces → underscores)
3. Trim string whitespace
4. Replace placeholders ("unknown", "?", 999) → NaN
5. Coerce numeric-looking object columns
6. Impute: median (numeric), mode (categorical)
7. Encode target to binary 0/1

### 2.2 Evaluation Pipeline

All models use scikit-learn Pipelines with `ColumnTransformer`:
- Numeric features: `StandardScaler`
- Categorical features: `OneHotEncoder(handle_unknown='ignore')`
- All transformations fitted only on training data — no leakage

CV strategy:
- Telco / Ecom: `StratifiedKFold(n_splits=5)`
- Bank: `TimeSeriesSplit(n_splits=5)` — temporal ordering prevents look-ahead bias
- Advanced models: nested CV (inner: hyperparameter tuning, outer: generalization)

---

## 3. Baseline Results

CV results (5-fold, mean ± std) from `evaluation.py`:

| Dataset | Model | Accuracy | F1 | ROC-AUC |
|---------|-------|----------|-----|---------|
| Telco | Logistic Regression | 80.19% ± 1.16% | 0.592 ± 0.030 | 0.846 ± 0.013 |
| Telco | k-NN (k=5) | 76.50% ± 0.71% | 0.547 ± 0.016 | 0.783 ± 0.007 |
| Telco | Naive Bayes | 69.65% ± 0.91% | 0.597 ± 0.010 | 0.821 ± 0.012 |
| Bank | Logistic Regression | 92.71% ± 1.80% | 0.475 ± 0.028 | 0.928 ± 0.018 |
| Bank | k-NN (k=5) | 92.40% ± 1.83% | 0.417 ± 0.057 | 0.838 ± 0.032 |
| Bank | Naive Bayes | 88.42% ± 2.69% | 0.350 ± 0.079 | 0.819 ± 0.047 |
| Ecom | Logistic Regression | 88.20% ± 0.49% | 0.497 ± 0.027 | 0.890 ± 0.014 |
| Ecom | k-NN (k=5) | 87.64% ± 0.55% | 0.496 ± 0.030 | 0.786 ± 0.011 |
| Ecom | Naive Bayes | 79.49% ± 0.66% | 0.484 ± 0.018 | 0.812 ± 0.009 |

**Key finding**: Logistic Regression dominates all three datasets among baselines. Naive
Bayes struggles on Bank due to high feature correlations (r=0.97) violating the independence
assumption. The Bank CV-test gap of ~5% is attributable to the temporal split (concept drift),
not overfitting.

---

## 4. Advanced Models

All metrics from nested cross-validation where available.

| Dataset | Model | ROC-AUC | Notes |
|---------|-------|---------|-------|
| Telco | MLP | 0.844 | Nested CV, StratifiedKFold |
| Telco | Random Forest | 0.838 | Nested CV, StratifiedKFold |
| Telco | SVM | — | GridSearchCV, accuracy only |
| Telco | Gradient Boosting | — | Nested CV, ROC-AUC |
| Bank | MLP | 0.941 | Nested CV, StratifiedKFold |
| **Bank** | **Random Forest** | **0.943** | **Best overall** |
| Bank | SVM | — | GridSearchCV, accuracy only |
| Bank | Gradient Boosting | — | Nested CV, ROC-AUC |
| Ecom | MLP | 0.916 | Nested CV, StratifiedKFold |
| Ecom | Random Forest | 0.924 | Nested CV, StratifiedKFold |
| Ecom | SVM | — | GridSearchCV, accuracy only |
| Ecom | Gradient Boosting | — | Nested CV, ROC-AUC |

**Best model**: Random Forest on Bank Marketing (ROC-AUC 0.943, nested CV, parameters:
n_estimators=200, max_depth=20, max_features='sqrt', class_weight='balanced').

---

## 5. Model Analysis — Best Model Deep-Dive

The `model_analysis.py` framework provides 16 model-agnostic functions (8 compute + 8 plot)
and is executed via `run_analysis.py` (standalone script) or `notebooks/model_analysis.ipynb`.

### 5.1 Analysis Components

| Analysis | Functions | Output |
|----------|-----------|--------|
| Permutation Importance | `compute_permutation_importance()` → `plot_permutation_importance()` | Bar chart: top 15 features by F1 decrease |
| SHAP | `compute_shap_values()` → `plot_shap_summary()` / `plot_shap_dependence()` | Beeswarm summary + dependence plots for top 3 features |
| Learning Curves | `compute_learning_curve()` → `plot_learning_curve()` | Train/val F1 vs. training set size |
| Confusion Matrix | `compute_confusion_analysis()` → `plot_confusion_matrix()` | Absolute + normalized 2x2 matrices |
| Hyperparameter Sensitivity | `analyze_hyperparameter_sensitivity()` → `plot_sensitivity_curve()` | CV score vs. parameter value |
| Misclassification Patterns | `analyze_misclassified_patterns()` → `plot_misclassification_diff()` | Feature means: correct vs. misclassified |

### 5.2 Running

```bash
# Full analysis on best model (Random Forest / Bank)
python src/run_analysis.py
```

Output: `reports/plots/*.png` (plots), `reports/results/analysis_results.csv` (metrics).

Edit the CONFIG block at the top of `run_analysis.py` to change dataset, model type, and
hyperparameters.

---

## 6. Methodology Notes

### Data Leakage Prevention
- All notebooks prior to the final version applied `pd.get_dummies` before train/test split.
  This is methodologically impure but not a true data leak (deterministic mapping, no learned
  statistics from the test set).
- The KNN notebook originally had critical leakage (`MinMaxScaler.fit_transform` +
  `SelectPercentile.fit_transform` on full data) — fixed via Pipeline-based preprocessing.
- `evaluation.py`, `gradient_boosting.py`, and `run_analysis.py` are leak-free
  (ColumnTransformer inside Pipeline).

### Metric Choice
- Accuracy is reported for completeness but F1 and ROC-AUC are the primary metrics
- Nested CV used for all advanced models with ROC-AUC as inner scoring
- StratifiedKFold preserves class distribution; TimeSeriesSplit for temporal Bank data

### Reproducibility
- Random state 42 throughout
- All scripts are idempotent (skip if output exists)
- `requirements.txt` pins all dependencies with exact versions

---

## 7. Project Structure

```
customer-decision-prediction/
├── .gitignore
├── README.md
├── doc.md                              # This file
├── requirements.txt
├── LICENSE
│
├── data/                               # .gitignored
│   ├── raw/
│   └── processed/
│
├── notebooks/                          # All Jupyter notebooks
│   ├── baseclassifier_comp.ipynb       # Baseline comparison notebook
│   ├── logreg_analysis.ipynb           # Logistic Regression
│   ├── knn_analysis.ipynb              # k-NN
│   ├── bayesfin.ipynb                  # Naive Bayes
│   ├── subvectormachine.ipynb          # SVM
│   ├── mlp_analysis.ipynb              # MLP
│   ├── rf_analysis.ipynb               # Random Forest
│   ├── gradient_boosting_analysis.ipynb
│   ├── Vergleich.ipynb                 # Cross-model comparison
│   └── model_analysis.ipynb            # Best model deep-dive
│
├── src/
│   ├── __init__.py
│   ├── setup_data.py                   # Data download + preprocessing
│   ├── analysis.py                     # Exploratory data analysis
│   ├── evaluation.py                   # Baseline evaluation (LR, KNN, NB)
│   ├── gradient_boosting.py            # Gradient Boosting evaluation
│   ├── model_analysis.py               # Analysis framework (16 functions)
│   ├── run_analysis.py                 # Standalone analysis runner
│   ├── inspect_time_structure.py       # Utility
│   └── utils/
│       ├── __init__.py
│       ├── download.py
│       └── preprocess.py
│
├── reports/                            # Generated outputs
│   ├── eda_report.txt
│   ├── summary_*.json
│   ├── plots/
│   │   ├── corr_heatmap_*.png
│   │   ├── permutation_imp_*.png
│   │   ├── shap_summary_*.png
│   │   ├── shap_dep_*.png
│   │   ├── learning_curve_*.png
│   │   ├── confusion_*.png
│   │   ├── misclass_diff_*.png
│   │   └── sensitivity_*.png
│   └── results/
│       ├── baseline_cv_results.csv
│       ├── baseline_test_results.csv
│       └── analysis_results.csv
│
└── vorlesung/                          # .gitignored — lecture materials
```

---

## 8. Usage Quick Reference

```bash
# First time: download and preprocess data
python src/setup_data.py

# Run exploratory data analysis (generates JSON summaries)
python src/analysis.py

# Baseline evaluation (Logistic Regression, k-NN, Naive Bayes)
python src/evaluation.py

# Best model deep-dive analysis
python src/run_analysis.py

# Interactive exploration
jupyter notebook notebooks/
```
