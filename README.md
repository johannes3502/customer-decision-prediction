# customer-decision-prediction

Comparative analysis of seven classifiers for predicting customer decisions across three
real-world tabular datasets — from linear baselines to tree ensembles and neural networks.

## Datasets

| Dataset | Rows | Features | Target | Minority Class |
|---------|------|----------|--------|----------------|
| Telco Customer Churn | 7,043 | 20 | Churn (y/n) | 26.5% |
| Bank Marketing | 41,176 | 20 | Term deposit (y/n) | 11.3% |
| Online Shoppers Intention | 12,205 | 17 | Purchase (y/n) | 15.6% |

All three datasets exhibit class imbalance, making F1 and ROC-AUC the primary evaluation
metrics. Accuracy alone is misleading for these problems.

## Key Results

All metrics from 5-fold cross-validation (nested CV for advanced models).

| Model | Telco F1 | Telco ROC-AUC | Bank F1 | Bank ROC-AUC | Ecom F1 | Ecom ROC-AUC |
|-------|----------|---------------|---------|--------------|---------|--------------|
| Logistic Regression | 0.592 | 0.846 | 0.475 | 0.928 | 0.497 | 0.890 |
| k-NN (k=5) | 0.547 | 0.783 | 0.417 | 0.838 | 0.496 | 0.786 |
| Naive Bayes | 0.597 | 0.821 | 0.350 | 0.819 | 0.484 | 0.812 |
| SVM | — | — | — | — | — | — |
| MLP | — | 0.844 | — | 0.941 | — | 0.916 |
| **Random Forest** | — | 0.838 | — | **0.943** | — | 0.924 |
| Gradient Boosting | — | — | — | — | — | — |

**Best model**: Random Forest on Bank Marketing (ROC-AUC 0.943, nested CV, parameters:
n_estimators=200, max_depth=20, max_features='sqrt', class_weight='balanced').

## Project Structure

```
├── data/                           # .gitignored — generated locally
│   ├── raw/                        # Downloaded CSVs
│   └── processed/                  # Cleaned CSVs (7-step pipeline)
├── notebooks/                      # All Jupyter notebooks
│   ├── baseclassifier_comp.ipynb   # Baseline comparison
│   ├── logreg_analysis.ipynb       # Logistic Regression
│   ├── knn_analysis.ipynb          # k-Nearest Neighbors
│   ├── bayesfin.ipynb              # Naive Bayes
│   ├── subvectormachine.ipynb      # Support Vector Machine
│   ├── mlp_analysis.ipynb          # Multi-Layer Perceptron
│   ├── rf_analysis.ipynb           # Random Forest
│   ├── gradient_boosting_analysis.ipynb
│   ├── Vergleich.ipynb             # Cross-model comparison
│   └── model_analysis.ipynb        # Best model deep-dive
├── src/
│   ├── setup_data.py               # Data download + preprocessing
│   ├── analysis.py                 # Exploratory data analysis
│   ├── evaluation.py               # Baseline evaluation (LR, KNN, NB)
│   ├── gradient_boosting.py        # Gradient Boosting evaluation
│   ├── model_analysis.py           # Analysis framework (16 functions)
│   ├── run_analysis.py             # Standalone analysis runner
│   ├── inspect_time_structure.py   # Temporal structure inspection
│   └── utils/
│       ├── download.py             # Dataset download utilities
│       └── preprocess.py           # 7-step preprocessing pipeline
├── reports/                        # Generated outputs
│   ├── eda_report.txt
│   ├── summary_*.json
│   ├── plots/                      # All generated figures
│   └── results/                    # Result CSV tables
├── doc.md                          # Full documentation
├── requirements.txt
└── LICENSE
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
python src/analysis.py         # Exploratory data analysis
python src/evaluation.py       # Baseline classifier comparison
python src/run_analysis.py     # Best model deep-dive analysis
```

Individual classifier notebooks in `notebooks/` can be explored interactively:

```bash
jupyter notebook notebooks/
```

## Methodology

- **Leak-free preprocessing**: All scaling and encoding inside scikit-learn Pipelines,
  fitted exclusively on training folds
- **Nested cross-validation** for advanced models: inner loop for hyperparameter tuning,
  outer loop for unbiased generalization estimates
- **StratifiedKFold** preserves class distribution; **TimeSeriesSplit** for the temporal
  Bank dataset
- **F1 and ROC-AUC** as primary metrics, appropriate for imbalanced classification
- Random seed 42 throughout for reproducibility

## License

See [LICENSE](LICENSE) for details. Dataset sources: Telco (IBM Community License),
Bank (CC BY 4.0), Ecom (UCI, public domain).
