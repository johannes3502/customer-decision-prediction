#!/usr/bin/env python3
"""
Task 4 – Model Analysis of the Best Performing Classifier
=========================================================

Standalone analysis script. Runs all 8 model analyses (Permutation Importance,
SHAP, Learning Curves, Confusion Matrix, Hyperparameter Sensitivity,
Misclassification Patterns) on the best model identified in Task 3.

Usage:
    python src/run_analysis.py

Configuration:
    Edit the CONFIG dict at the top of this script to select the dataset,
    model type, and hyperparameters before running.

Output:
    Plots saved to reports/plots/*.png
    Results saved to reports/results/analysis_results.csv
"""

import sys
from pathlib import Path
import json
import logging
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import evaluation
from model_analysis import (
    compute_permutation_importance,
    compute_shap_values,
    compute_learning_curve,
    compute_confusion_analysis,
    analyze_hyperparameter_sensitivity,
    analyze_misclassified_patterns,
    plot_permutation_importance,
    plot_shap_summary,
    plot_shap_dependence,
    plot_learning_curve,
    plot_confusion_matrix,
    plot_sensitivity_curve,
    plot_misclassification_diff,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_analysis')

RANDOM_STATE = 42
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION BLOCK                                ║
# ║  EDIT THIS SECTION based on Task 3 comparison results.                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CONFIG = {
    # Dataset: 'telco' | 'bank' | 'ecom'
    'dataset': 'bank',

    # Model display name (for plot titles / filenames)
    'model_name': 'Random Forest',

    # scikit-learn classifier class + hyperparameters
    # Import the class below and set its constructor kwargs here.
    # Some examples (commented out):

    # --- Random Forest (highest ROC-AUC on Bank: 0.943) ---
    'classifier_class': 'sklearn.ensemble.RandomForestClassifier',
    'classifier_kwargs': {
        'n_estimators': 200,
        'max_depth': 20,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    },

    # --- MLP (close second on Bank: 0.941) ---
    # 'classifier_class': 'sklearn.neural_network.MLPClassifier',
    # 'classifier_kwargs': {
    #     'hidden_layer_sizes': (128, 64),
    #     'alpha': 0.001,
    #     'early_stopping': True,
    #     'random_state': RANDOM_STATE,
    #     'max_iter': 1000,
    # },

    # --- Gradient Boosting ---
    # 'classifier_class': 'sklearn.ensemble.HistGradientBoostingClassifier',
    # 'classifier_kwargs': {
    #     'max_iter': 300,
    #     'learning_rate': 0.1,
    #     'max_depth': 15,
    #     'random_state': RANDOM_STATE,
    # },

    # --- SVM ---
    # 'classifier_class': 'sklearn.svm.SVC',
    # 'classifier_kwargs': {
    #     'C': 10,
    #     'kernel': 'rbf',
    #     'probability': True,
    #     'random_state': RANDOM_STATE,
    # },

    # Hyperparameter Sensitivity Analysis: which param(s) to vary?
    # Format: {param_name: [value1, value2, ...]}
    'sensitivity_params': {
        'n_estimators': [10, 25, 50, 100, 200, 300, 500],
        'max_depth': [3, 5, 10, 15, 20, None],
    },

    # SHAP samples (more = slower but more stable)
    'shap_max_samples': 200,

    # Permutation importance repeats
    'permutation_n_repeats': 10,

    # Scoring metric (f1 is correct for imbalanced data)
    'scoring': 'f1',
}


def import_class(full_name):
    """Dynamically import a class from a dotted path string."""
    mod_name, cls_name = full_name.rsplit('.', 1)
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)


def build_pipeline(numeric_features, categorical_features, classifier_class, classifier_kwargs):
    """Build a sklearn Pipeline with proper preprocessing.

    Uses the same preprocessor strategy as evaluation.py (StandardScaler for
    numeric, OneHotEncoder for categorical) to ensure no data leakage.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ],
        remainder='drop',
    )

    classifier = classifier_class(**classifier_kwargs)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier),
    ])


def main():
    cfg = CONFIG
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("MODEL ANALYSIS")
    logger.info(f"Dataset: {cfg['dataset']}   Model: {cfg['model_name']}")
    logger.info("=" * 70)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    logger.info("\n[1/7] Loading data...")
    dataset_config = [d for d in evaluation.DATASETS if d['name'] == cfg['dataset']]
    if not dataset_config:
        raise ValueError(f"Unknown dataset '{cfg['dataset']}'. Choose: telco, bank, ecom")
    dataset_config = dataset_config[0]

    eda_summary = evaluation.load_eda_summary(cfg['dataset'])
    X, y, numeric_features, categorical_features = evaluation.load_and_prepare_data(
        dataset_config, eda_summary, dataset_config['split_mode']
    )

    # Train/Test split
    split_mode = dataset_config['split_mode']
    if split_mode == 'temporal':
        from sklearn.model_selection import TimeSeriesSplit
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        cv = TimeSeriesSplit(n_splits=5)
    else:
        from sklearn.model_selection import train_test_split, StratifiedKFold
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    logger.info(f"  Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    logger.info(f"  Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")
    logger.info(f"  Class balance (train): 0={sum(y_train == 0)}, 1={sum(y_train == 1)}")

    # ── 2. Build & Train Pipeline ─────────────────────────────────────────
    logger.info("\n[2/7] Building and training pipeline...")
    classifier_class = import_class(cfg['classifier_class'])
    pipeline = build_pipeline(
        numeric_features, categorical_features,
        classifier_class, cfg['classifier_kwargs']
    )
    pipeline.fit(X_train, y_train)

    # Quick test score
    y_pred = pipeline.predict(X_test)
    from sklearn.metrics import f1_score, roc_auc_score
    test_f1 = f1_score(y_test, y_pred)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    test_roc = roc_auc_score(y_test, y_proba)
    logger.info(f"  Test F1: {test_f1:.4f}   Test ROC-AUC: {test_roc:.4f}")

    # ── 3. Permutation Importance ─────────────────────────────────────────
    logger.info("\n[3/7] Permutation Importance...")
    feature_names = X.columns.tolist()
    imp_df = compute_permutation_importance(
        pipeline, X_test, y_test, feature_names,
        n_repeats=cfg['permutation_n_repeats'], scoring=cfg['scoring']
    )
    plot_permutation_importance(imp_df, cfg['dataset'], cfg['model_name'], top_n=15)
    plt.close('all')
    logger.info(f"  Top 5 features:\n{imp_df.head(5).to_string(index=False)}")

    # ── 4. SHAP Analysis ──────────────────────────────────────────────────
    logger.info("\n[4/7] SHAP Analysis...")
    shap_vals, explainer, X_trans, encoded_names = compute_shap_values(
        pipeline, X_test, max_samples=cfg['shap_max_samples']
    )
    if shap_vals is not None:
        plot_shap_summary(shap_vals, X_trans, encoded_names, cfg['dataset'], cfg['model_name'])
        plt.close('all')

        # Dependence plots for top 3 permutation features
        top_features_raw = imp_df['feature'].head(3).tolist()
        top_encoded = []
        for feat in top_features_raw:
            if feat in numeric_features:
                top_encoded.append((f'num__{feat}', feat))
            elif feat in categorical_features:
                matching = [n for n in encoded_names if n.startswith(f'cat__{feat}_')]
                if matching:
                    top_encoded.append((matching[0], feat))
                else:
                    logger.warning(f"  No encoded column found for categorical feature '{feat}'")
        for feat_encoded, feat_raw in top_encoded:
            try:
                plot_shap_dependence(
                    shap_vals, X_trans, encoded_names, feat_encoded,
                    cfg['dataset'], cfg['model_name'],
                    display_name=feat_raw,
                )
                plt.close('all')
            except Exception as e:
                logger.warning(f"  SHAP dependence plot for '{feat_raw}' failed: {e}")
    else:
        logger.warning("  SHAP unavailable – skipping SHAP plots.")

    # ── 5. Hyperparameter Sensitivity ─────────────────────────────────────
    logger.info("\n[5/7] Hyperparameter Sensitivity...")
    for param_name, param_values in cfg['sensitivity_params'].items():
        # Skip params not applicable to this classifier
        if param_name not in classifier_class().get_params():
            logger.warning(f"  Skipping '{param_name}' – not a valid param for {classifier_class.__name__}")
            continue

        result = analyze_hyperparameter_sensitivity(
            classifier_class, param_name, param_values,
            pipeline, X_train, y_train, cv, scoring=cfg['scoring']
        )
        plot_sensitivity_curve(
            result['param_values'], result['mean_scores'], result['std_scores'],
            param_name, cfg['dataset'], cfg['model_name'], scoring=cfg['scoring']
        )
        plt.close('all')
        logger.info(f"  Best {param_name}: {result['best_value']} ({result['best_score']:.4f})")

    # ── 6. Learning Curves ────────────────────────────────────────────────
    logger.info("\n[6/7] Learning Curves...")
    lc = compute_learning_curve(pipeline, X_train, y_train, cv, scoring=cfg['scoring'])
    plot_learning_curve(
        lc['train_sizes'], lc['train_mean'], lc['train_std'],
        lc['val_mean'], lc['val_std'], cfg['dataset'], cfg['model_name'],
        scoring=cfg['scoring']
    )
    plt.close('all')
    gap = lc['train_mean'][-1] - lc['val_mean'][-1]
    logger.info(f"  Bias/Variance Gap: {gap:.4f}  (train={lc['train_mean'][-1]:.4f}, val={lc['val_mean'][-1]:.4f})")

    # ── 7. Error Analysis ─────────────────────────────────────────────────
    logger.info("\n[7/7] Error Analysis...")

    # Confusion Matrix
    ca = compute_confusion_analysis(pipeline, X_test, y_test)
    plot_confusion_matrix(ca['cm'], cfg['dataset'], cfg['model_name'])
    plt.close('all')
    logger.info(
        f"  Precision: {ca['precision']:.4f}  Recall: {ca['recall']:.4f}  "
        f"F1: {ca['f1']:.4f}  Specificity: {ca['specificity']:.4f}"
    )
    logger.info(
        f"  Misclassified: {len(ca['misclassified_idx'])}/{len(y_test)} "
        f"({len(ca['misclassified_idx'])/len(y_test)*100:.1f}%)"
    )

    # Misclassification Pattern Analysis
    diff_df = analyze_misclassified_patterns(pipeline, X_test, y_test, feature_names)
    plot_misclassification_diff(diff_df, cfg['dataset'], cfg['model_name'], top_n=10)
    plt.close('all')
    logger.info(f"  Top misclass feature diff: {diff_df.iloc[0]['feature']} (abs_diff={diff_df.iloc[0]['abs_diff']:.4f})")

    # ── Save Results Summary ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS...")

    results_path = evaluation.REPORTS_DIR / 'results' / 'analysis_results.csv'
    results = {
        'dataset': cfg['dataset'],
        'model': cfg['model_name'],
        'test_f1': round(test_f1, 4),
        'test_roc_auc': round(test_roc, 4),
        'precision': round(ca['precision'], 4),
        'recall': round(ca['recall'], 4),
        'specificity': round(ca['specificity'], 4),
        'misclassification_rate': round(len(ca['misclassified_idx']) / len(y_test), 4),
        'bias_variance_gap': round(gap, 4),
        'top_permutation_feature': imp_df.iloc[0]['feature'],
        'top_permutation_importance': round(imp_df.iloc[0]['importance_mean'], 4),
    }
    pd.DataFrame([results]).to_csv(results_path, index=False)
    logger.info(f"  Results saved to {results_path}")

    # Log top features
    logger.info(f"\nTop 10 Features (Permutation Importance):")
    for _, row in imp_df.head(10).iterrows():
        logger.info(f"  {row['feature']:35s}  {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 70}")
    logger.info(f"ANALYSIS COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Plots saved to: {evaluation.PROJECT_ROOT / 'reports' / 'plots'}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
