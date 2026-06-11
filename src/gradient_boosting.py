"""
Gradient Boosting evaluation with nested cross-validation.

Uses the same data pipeline and preprocessing as evaluation.py (StandardScaler +
OneHotEncoder) to ensure fair comparison. Nested CV follows the pattern established
in the MLP and Random Forest notebooks: GridSearchCV for hyperparameter tuning
inside StratifiedKFold / TimeSeriesSplit for unbiased generalization estimates.

Scoring: ROC-AUC (inner loop), F1 + ROC-AUC (outer loop, test set).

Usage: python src/gradient_boosting.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, GridSearchCV, cross_val_score, train_test_split
)
from sklearn.metrics import f1_score, roc_auc_score

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from evaluation import (
    load_eda_summary,
    load_and_prepare_data,
    DATASETS,
    DATA_PROCESSED_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gradient_boosting")

RANDOM_STATE = 42
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"

NESTED_CV_OUTER = 5
NESTED_CV_INNER = 5
TEST_SIZE = 0.2

PARAM_GRID = {
    "model__max_iter": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [5, 10, 15, None],
}


def build_gb_pipeline(numeric_features, categorical_features):
    """Build Gradient Boosting Pipeline with StandardScaler + OneHotEncoder.

    Uses the identical preprocessor as evaluation.py to ensure fair comparison.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        early_stopping=False,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate_gb_dataset(dataset_config, eda_summary, split_mode):
    """Run nested cross-validation and test-set evaluation for one dataset.

    Returns dict with nested CV scores, test metrics, and best parameters.
    """
    dataset_name = dataset_config["name"]
    logger.info(f"\n{'=' * 70}")
    logger.info(f"GRADIENT BOOSTING: {dataset_name.upper()} (split_mode={split_mode})")
    logger.info(f"{'=' * 70}")

    X, y, numeric_features, categorical_features = load_and_prepare_data(
        dataset_config, eda_summary, split_mode
    )

    pipeline = build_gb_pipeline(numeric_features, categorical_features)

    # Train / test split (consistent with evaluation.py)
    if split_mode == "temporal":
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        logger.info(f"{dataset_name}: Temporal split at index {split_idx}.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

    logger.info(
        f"{dataset_name}: Training set {X_train.shape[0]} rows, "
        f"Test set {X_test.shape[0]} rows"
    )

    # Cross-validation strategy
    if split_mode == "temporal":
        outer_cv = TimeSeriesSplit(n_splits=NESTED_CV_OUTER)
    else:
        outer_cv = StratifiedKFold(
            n_splits=NESTED_CV_OUTER, shuffle=True, random_state=RANDOM_STATE
        )
    inner_cv = StratifiedKFold(
        n_splits=NESTED_CV_INNER, shuffle=True, random_state=RANDOM_STATE
    )

    # Inner search: grid over hyperparameters, scored by ROC-AUC
    inner_search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    logger.info(f"{dataset_name}: Running nested CV (outer={NESTED_CV_OUTER}, inner={NESTED_CV_INNER})...")
    nested_scores = cross_val_score(
        inner_search, X_train, y_train, cv=outer_cv, scoring="roc_auc", n_jobs=-1
    )

    nested_mean = float(nested_scores.mean())
    nested_std = float(nested_scores.std())
    logger.info(
        f"{dataset_name}: Nested CV ROC-AUC: {nested_mean:.4f} \u00b1 {nested_std:.4f}"
    )
    logger.info(
        f"  Fold scores: {[round(s, 4) for s in nested_scores.tolist()]}"
    )

    # Refit inner search on full training set to extract best params and evaluate
    inner_search.fit(X_train, y_train)
    best_params = {
        k.replace("model__", ""): v
        for k, v in inner_search.best_params_.items()
    }
    logger.info(f"{dataset_name}: Best params: {best_params}")

    # Test set evaluation
    best_pipeline = inner_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_f1 = float(f1_score(y_test, y_pred))
    test_roc_auc = float(roc_auc_score(y_test, y_proba))
    logger.info(
        f"{dataset_name}: Test F1={test_f1:.4f}  ROC-AUC={test_roc_auc:.4f}"
    )

    return {
        "dataset": dataset_name,
        "split_mode": split_mode,
        "nested_cv_roc_auc_mean": nested_mean,
        "nested_cv_roc_auc_std": nested_std,
        "nested_cv_fold_scores": [round(s, 4) for s in nested_scores.tolist()],
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "best_params": best_params,
    }


def main():
    """Run Gradient Boosting evaluation on all three datasets."""
    logger.info("=" * 70)
    logger.info("GRADIENT BOOSTING — NESTED CROSS-VALIDATION")
    logger.info("=" * 70)

    all_results = []

    for dataset_config in DATASETS:
        dataset_name = dataset_config["name"]
        split_mode = dataset_config["split_mode"]

        eda_summary = load_eda_summary(dataset_name)
        result = evaluate_gb_dataset(dataset_config, eda_summary, split_mode)
        all_results.append(result)

    # Compile and save results
    rows = []
    for r in all_results:
        rows.append({
            "Dataset": r["dataset"],
            "Split Mode": r["split_mode"],
            "Nested CV ROC-AUC (mean)": round(r["nested_cv_roc_auc_mean"], 4),
            "Nested CV ROC-AUC (std)": round(r["nested_cv_roc_auc_std"], 4),
            "Test F1": round(r["test_f1"], 4),
            "Test ROC-AUC": round(r["test_roc_auc"], 4),
            "Best Params": str(r["best_params"]),
        })

    df_results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "gradient_boosting_results.csv"
    df_results.to_csv(output_path, index=False)

    logger.info(f"\n{'=' * 90}")
    logger.info("GRADIENT BOOSTING RESULTS")
    logger.info(f"{'=' * 90}")
    print(df_results.to_string(index=False))
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"{'=' * 90}")


if __name__ == "__main__":
    main()
