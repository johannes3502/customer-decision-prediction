"""
Model analysis for Task 4 – investigates the best-performing classifier.

Provides model-agnostic functions for feature importance (Permutation + SHAP),
learning curves, confusion matrix analysis, hyperparameter sensitivity, and
error pattern analysis.

All functions accept sklearn Pipeline objects (ColumnTransformer + Classifier)
and numpy/pandas data arrays. Plot functions save to reports/plots/.

Usage: from src.model_analysis import compute_permutation_importance, ...
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

_TREE_CLASSES = None


def _get_tree_classes():
    global _TREE_CLASSES
    if _TREE_CLASSES is None:
        try:
            from sklearn.ensemble import (
                RandomForestClassifier,
                GradientBoostingClassifier,
                HistGradientBoostingClassifier,
                ExtraTreesClassifier,
            )
            from sklearn.tree import DecisionTreeClassifier

            _TREE_CLASSES = (
                RandomForestClassifier,
                GradientBoostingClassifier,
                HistGradientBoostingClassifier,
                ExtraTreesClassifier,
                DecisionTreeClassifier,
            )
        except ImportError:
            _TREE_CLASSES = ()
    return _TREE_CLASSES


def _is_tree_based(classifier):
    return isinstance(classifier, _get_tree_classes())


def _ensure_numpy(X):
    if hasattr(X, 'values'):
        return X.values
    return np.asarray(X)





# ---------------------------------------------------------------------------
# 1. Feature Importance – Permutation
# ---------------------------------------------------------------------------

def compute_permutation_importance(pipeline, X_test, y_test, feature_names,
                                   n_repeats=10, scoring='f1'):
    """Compute permutation importance for a fitted sklearn Pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline (preprocessor + classifier).
        X_test: pandas DataFrame with raw features.
        y_test: Target array or Series.
        feature_names: List of original feature names (before encoding).
        n_repeats: Number of shuffling repetitions per feature.
        scoring: Metric name (default 'f1' for imbalanced data).

    Returns:
        pd.DataFrame with columns [feature, importance_mean, importance_std],
        sorted descending by importance_mean.
    """
    logger.info(f"Computing permutation importance (n_repeats={n_repeats}, scoring={scoring})...")
    r = permutation_importance(
        pipeline, X_test, y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': r.importances_mean,
        'importance_std': r.importances_std,
    })
    df = df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    logger.info(f"Top feature: {df.iloc[0]['feature']} ({df.iloc[0]['importance_mean']:.4f})")
    return df


def plot_permutation_importance(importance_df, dataset_name, model_name,
                                top_n=15, save=True):
    """Plot permutation importance as horizontal bar chart.

    Args:
        importance_df: DataFrame from compute_permutation_importance().
        dataset_name: Name of the dataset (for title and filename).
        model_name: Name of the model (for title and filename).
        top_n: Show only top N features.
        save: If True, save to PLOTS_DIR.
    """
    df_plot = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.3)))
    ax.barh(df_plot['feature'], df_plot['importance_mean'],
            xerr=df_plot['importance_std'], capsize=3, color='steelblue')
    ax.set_xlabel('Permutation Importance (F1 decrease)', fontsize=11)
    ax.set_title(f'Permutation Importance – {model_name}\n{dataset_name}', fontsize=13)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f'permutation_imp_{dataset_name}_{model_name.replace(" ", "_").lower()}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved permutation importance plot to {path}")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Feature Importance – SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(pipeline, X_sample, max_samples=200):
    """Compute SHAP values for a fitted sklearn Pipeline.

    Automatically selects TreeExplainer for tree-based classifiers
    and KernelExplainer as fallback for all other models.

    Args:
        pipeline: Fitted sklearn Pipeline (preprocessor + classifier).
        X_sample: pandas DataFrame with raw features (background sample).
        max_samples: Maximum samples for KernelExplainer (ignored for trees).

    Returns:
        Tuple (shap_values, explainer, X_trans, encoded_feature_names) or
        (None, None, None, None) if SHAP is unavailable or fails.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None, None, None, None

    classifier = pipeline.named_steps.get('classifier')
    preprocessor = pipeline.named_steps.get('preprocessor')

    if classifier is None:
        logger.error("Pipeline has no 'classifier' step.")
        return None, None, None, None

    X_trans = preprocessor.transform(X_sample) if preprocessor else _ensure_numpy(X_sample)
    encoded_names = (
        preprocessor.get_feature_names_out().tolist()
        if preprocessor and hasattr(preprocessor, 'get_feature_names_out')
        else list(range(X_trans.shape[1]))
    )
    encoded_names = [str(n) for n in encoded_names]

    logger.info(f"Computing SHAP values (model type: {type(classifier).__name__})...")

    try:
        if _is_tree_based(classifier):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_trans)

            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]
            logger.info("SHAP TreeExplainer completed.")
        else:
            background = X_trans[:min(50, X_trans.shape[0])]
            to_explain = X_trans[:min(max_samples, X_trans.shape[0])]

            predict_fn = (
                classifier.predict_proba
                if hasattr(classifier, 'predict_proba')
                else classifier.predict
            )

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(to_explain)

            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]

            logger.info("SHAP KernelExplainer completed.")
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}. Returning None.")
        return None, None, None, None

    return shap_values, explainer, X_trans, encoded_names


def plot_shap_summary(shap_values, X_trans, feature_names, dataset_name,
                      model_name, save=True):
    """Plot SHAP beeswarm summary plot.

    Args:
        shap_values: SHAP values array from compute_shap_values().
        X_trans: Preprocessed feature matrix.
        feature_names: List of feature names after encoding.
        dataset_name: Dataset name.
        model_name: Model name.
        save: If True, save to PLOTS_DIR.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed.")
        return

    shap.summary_plot(shap_values, X_trans, feature_names=feature_names,
                      show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle(f'SHAP Summary – {model_name}\n{dataset_name}', fontsize=13)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f'shap_summary_{dataset_name}_{model_name.replace(" ", "_").lower()}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {path}")
    plt.show()


def plot_shap_dependence(shap_values, X_trans, feature_names, target_feature,
                          dataset_name, model_name, interaction_feature=None,
                          save=True, display_name=None):
    """Plot SHAP dependence plot for a single feature.

    Args:
        shap_values: SHAP values array.
        X_trans: Preprocessed feature matrix.
        feature_names: List of feature names after encoding.
        target_feature: Name or index of the feature to plot.
        dataset_name: Dataset name.
        model_name: Model name.
        interaction_feature: Optional interaction feature name/index for coloring.
        save: If True, save to PLOTS_DIR.
        display_name: Optional display name for the feature (uses encoded name if None).
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed.")
        return

    if isinstance(target_feature, str):
        try:
            feature_idx = feature_names.index(target_feature)
        except ValueError:
            logger.error(f"Feature '{target_feature}' not found in feature_names.")
            return
    else:
        feature_idx = target_feature

    interaction_idx = None
    if interaction_feature is not None:
        if isinstance(interaction_feature, str):
            try:
                interaction_idx = feature_names.index(interaction_feature)
            except ValueError:
                logger.warning(f"Interaction feature '{interaction_feature}' not found.")
                interaction_idx = None
        else:
            interaction_idx = interaction_feature

    shap.dependence_plot(
        feature_idx, shap_values, X_trans,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        show=False,
    )
    fig = plt.gcf()
    feat_display = display_name if display_name else (feature_names[feature_idx] if feature_idx < len(feature_names) else str(feature_idx))
    fig.suptitle(f'SHAP Dependence: {feat_display} – {model_name}\n{dataset_name}', fontsize=13)
    plt.tight_layout()
    if save:
        safe_feat = str(feat_display).replace('/', '_').replace(' ', '_')[:50]
        path = PLOTS_DIR / f'shap_dep_{dataset_name}_{safe_feat}_{model_name.replace(" ", "_").lower()}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved SHAP dependence plot to {path}")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Learning Curves
# ---------------------------------------------------------------------------

def compute_learning_curve(pipeline, X_train, y_train, cv, scoring='f1'):
    """Compute learning curve for a model.

    Args:
        pipeline: sklearn Pipeline (will be re-fitted for each train size).
        X_train: Raw training features.
        y_train: Training target.
        cv: CV splitter (StratifiedKFold or TimeSeriesSplit).
        scoring: Metric name.

    Returns:
        dict with keys: train_sizes, train_mean, train_std, val_mean, val_std.
    """
    logger.info(f"Computing learning curve (scoring={scoring})...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    train_sizes_pct = (train_sizes_abs / len(X_train)) * 100

    result = {
        'train_sizes': train_sizes_pct,
        'train_sizes_abs': train_sizes_abs,
        'train_mean': train_scores.mean(axis=1),
        'train_std': train_scores.std(axis=1),
        'val_mean': val_scores.mean(axis=1),
        'val_std': val_scores.std(axis=1),
    }

    gap = result['train_mean'][-1] - result['val_mean'][-1]
    logger.info(f"Learning curve gap at full data: {gap:.4f} "
                f"(train={result['train_mean'][-1]:.4f}, val={result['val_mean'][-1]:.4f})")
    return result


def plot_learning_curve(train_sizes, train_mean, train_std, val_mean, val_std,
                        dataset_name, model_name, scoring='f1', save=True):
    """Plot learning curve with training and validation scores.

    Args:
        train_sizes: Array of training sizes (percentage).
        train_mean, train_std: Training score arrays.
        val_mean, val_std: Validation score arrays.
        dataset_name: Dataset name.
        model_name: Model name.
        scoring: Metric name for axis label.
        save: If True, save to PLOTS_DIR.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='steelblue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='darkorange')
    ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training Score')
    ax.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation Score')
    ax.set_xlabel('Training Set Size (%)', fontsize=11)
    ax.set_ylabel(f'{scoring.upper()} Score', fontsize=11)
    ax.set_title(f'Learning Curve – {model_name}\n{dataset_name}', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f'learning_curve_{dataset_name}_{model_name.replace(" ", "_").lower()}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved learning curve to {path}")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Confusion Matrix & Error Analysis
# ---------------------------------------------------------------------------

def compute_confusion_analysis(pipeline, X_test, y_test):
    """Compute confusion matrix, per-class metrics, and misclassified indices.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test features.
        y_test: Test target.

    Returns:
        dict with keys: cm, tn, fp, fn, tp, precision, recall, specificity,
                        misclassified_idx, y_pred.
    """
    y_pred = pipeline.predict(X_test)
    y_test_arr = _ensure_numpy(y_test)
    cm = confusion_matrix(y_test_arr, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    misclassified_idx = np.where(y_pred != y_test_arr)[0]

    logger.info(
        f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp} | "
        f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Specificity={specificity:.4f} | "
        f"Misclassified: {len(misclassified_idx)}/{len(y_test_arr)} "
        f"({len(misclassified_idx)/len(y_test_arr)*100:.1f}%)"
    )

    return {
        'cm': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'misclassified_idx': misclassified_idx,
        'y_pred': y_pred,
    }


def plot_confusion_matrix(cm, dataset_name, model_name, save=True):
    """Plot absolute and normalized confusion matrices side by side.

    Args:
        cm: 2x2 confusion matrix array.
        dataset_name: Dataset name.
        model_name: Model name.
        save: If True, save to PLOTS_DIR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ['Negative', 'Positive']

    disp_abs = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp_abs.plot(ax=axes[0], cmap='Blues', colorbar=False, values_format='d')
    axes[0].set_title(f'{model_name}\n(Absolute)', fontsize=12)

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_norm, row_sums,
        out=np.zeros_like(cm_norm, dtype=float),
        where=row_sums != 0,
    )
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp_norm.plot(ax=axes[1], cmap='Blues', colorbar=False, values_format='.2f')
    axes[1].set_title(f'{model_name}\n(Normalized)', fontsize=12)

    fig.suptitle(f'Confusion Matrix – {dataset_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f'confusion_{dataset_name}_{model_name.replace(" ", "_").lower()}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {path}")
    plt.show()


def analyze_misclassified_patterns(pipeline, X_test, y_test, feature_names):
    """Compare correctly vs incorrectly classified samples statistically.

    Transforms all features through the pipeline preprocessor to obtain
    numerical representations (including encoded categoricals), then
    computes the mean value for correct and misclassified groups and
    ranks by absolute difference.

    Args:
        pipeline: Fitted sklearn Pipeline with 'preprocessor' step.
        X_test: Raw test features (pandas DataFrame).
        y_test: Test target.
        feature_names: Original feature names (for fallback column naming).

    Returns:
        pd.DataFrame with columns [feature, correct_mean, misclassified_mean,
        abs_diff], sorted descending by abs_diff.
    """
    y_pred = pipeline.predict(X_test)
    y_test_arr = np.asarray(y_test).ravel()
    correct_mask = (y_pred == y_test_arr)
    incorrect_mask = ~correct_mask

    preprocessor = pipeline.named_steps.get('preprocessor')
    if preprocessor is not None and hasattr(preprocessor, 'transform'):
        X_trans = preprocessor.transform(X_test)
        encoded_names = (
            preprocessor.get_feature_names_out().tolist()
            if hasattr(preprocessor, 'get_feature_names_out')
            else [f'f{i}' for i in range(X_trans.shape[1])]
        )
        encoded_names = [str(n) for n in encoded_names]
        X_num = pd.DataFrame(X_trans, columns=encoded_names)
    else:
        X_num = X_test.select_dtypes(include=[np.number])
        if X_num.empty:
            logger.warning("No numeric features and no preprocessor available. "
                           "Falling back to all columns with coercion.")
            X_num = X_test.apply(pd.to_numeric, errors='coerce')

    results = []
    for col in X_num.columns:
        col_data = X_num[col].values.astype(float)
        correct_mean = np.nanmean(col_data[correct_mask]) if correct_mask.sum() > 0 else np.nan
        misclassified_mean = np.nanmean(col_data[incorrect_mask]) if incorrect_mask.sum() > 0 else np.nan
        abs_diff = abs(correct_mean - misclassified_mean) if not (np.isnan(correct_mean) or np.isnan(misclassified_mean)) else 0.0
        results.append({
            'feature': col,
            'correct_mean': correct_mean,
            'misclassified_mean': misclassified_mean,
            'abs_diff': abs_diff,
        })

    df = pd.DataFrame(results).sort_values('abs_diff', ascending=False).reset_index(drop=True)
    if len(df) > 0:
        logger.info(f"Top misclassification pattern: feature='{df.iloc[0]['feature']}' (abs_diff={df.iloc[0]['abs_diff']:.4f})")
    return df


def plot_misclassification_diff(diff_df, dataset_name, model_name, top_n=10, save=True):
    """Plot horizontal bar chart of feature mean differences between
    correctly and incorrectly classified samples.

    Args:
        diff_df: DataFrame from analyze_misclassified_patterns().
        dataset_name: Dataset name.
        model_name: Model name.
        top_n: Show top N features with largest absolute difference.
        save: If True, save to PLOTS_DIR.
    """
    df_plot = diff_df.head(top_n).iloc[::-1].copy()
    df_plot['correct_mean'] = df_plot['correct_mean'].fillna(0)
    df_plot['misclassified_mean'] = df_plot['misclassified_mean'].fillna(0)

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.3)))
    x = np.arange(len(df_plot))
    width = 0.35
    ax.barh(x + width / 2, df_plot['correct_mean'], width,
            label='Correct', color='steelblue', alpha=0.85)
    ax.barh(x - width / 2, df_plot['misclassified_mean'], width,
            label='Misclassified', color='darkorange', alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(df_plot['feature'])
    ax.set_xlabel('Mean Feature Value', fontsize=11)
    ax.set_title(f'Feature Differences: Correct vs Misclassified – {model_name}\n{dataset_name}', fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f'misclass_diff_{dataset_name}_{model_name.replace(" ", "_").lower()}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved misclassification diff plot to {path}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Hyperparameter Sensitivity
# ---------------------------------------------------------------------------

def analyze_hyperparameter_sensitivity(model_class, param_name, param_values,
                                        pipeline, X_train, y_train, cv,
                                        scoring='f1'):
    """Evaluate how a single hyperparameter affects model performance.

    Clones the provided pipeline and replaces only the classifier's
    target parameter for each value in param_values. Cross-validation
    scores are computed on the training set.

    Args:
        model_class: sklearn classifier class (e.g., RandomForestClassifier).
        param_name: Name of the parameter to vary.
        param_values: List of values to test.
        pipeline: Fitted or unfitted sklearn Pipeline whose classifier step
                  will be replaced during sensitivity analysis.
        X_train: Training features.
        y_train: Training target.
        cv: CV splitter.
        scoring: Metric name.

    Returns:
        dict with keys: param_values, mean_scores, std_scores, best_value, best_score.
    """
    logger.info(f"Hyperparameter sensitivity: {param_name} in {param_values} (scoring={scoring})...")

    mean_scores = []
    std_scores = []

    for val in param_values:
        try:
            pipe_clone = clone(pipeline)
            classifier_step = model_class(random_state=RANDOM_STATE)
            classifier_step.set_params(**{param_name: val})
            pipe_clone.steps[-1] = ('classifier', classifier_step)

            cv_scores = cross_val_score(
                pipe_clone, X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=-1,
            )
            mean_scores.append(cv_scores.mean())
            std_scores.append(cv_scores.std())
            logger.info(f"  {param_name}={val}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            logger.warning(f"  {param_name}={val}: FAILED ({e}). Setting NaN.")
            mean_scores.append(np.nan)
            std_scores.append(np.nan)

    best_idx = np.nanargmax(mean_scores) if not all(np.isnan(mean_scores)) else 0
    return {
        'param_values': param_values,
        'mean_scores': np.array(mean_scores),
        'std_scores': np.array(std_scores),
        'best_value': param_values[best_idx],
        'best_score': mean_scores[best_idx],
    }


def plot_sensitivity_curve(param_values, mean_scores, std_scores, param_name,
                           dataset_name, model_name, scoring='f1', save=True):
    """Plot hyperparameter sensitivity curve.

    Args:
        param_values: Tested parameter values.
        mean_scores: Mean CV scores.
        std_scores: Standard deviations.
        param_name: Name of the parameter.
        dataset_name: Dataset name.
        model_name: Model name.
        scoring: Metric name.
        save: If True, save to PLOTS_DIR.
    """
    valid = ~np.isnan(mean_scores)
    if not valid.any():
        logger.warning("All sensitivity scores are NaN. Cannot plot.")
        return

    param_vals = np.array(param_values)[valid]
    means = np.array(mean_scores)[valid]
    stds = np.array(std_scores)[valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        range(len(param_vals)),
        means - stds,
        means + stds,
        alpha=0.2,
        color='steelblue',
    )
    ax.plot(range(len(param_vals)), means, 'o-', color='steelblue', linewidth=2)

    best_idx = np.argmax(means)
    ax.axvline(best_idx, color='darkorange', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.scatter([best_idx], [means[best_idx]], color='darkorange', s=100, zorder=5,
               label=f'Best: {param_vals[best_idx]} ({means[best_idx]:.4f})')

    ax.set_xticks(range(len(param_vals)))
    ax.set_xticklabels([str(v) for v in param_vals], rotation=45, ha='right')
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel(f'{scoring.upper()} Score (CV)', fontsize=11)
    ax.set_title(f'Hyperparameter Sensitivity: {param_name} – {model_name}\n{dataset_name}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        safe_param = str(param_name).replace('/', '_').replace(' ', '_')
        path = PLOTS_DIR / f'sensitivity_{safe_param}_{dataset_name}_{model_name.replace(" ", "_").lower()}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved sensitivity curve to {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'compute_permutation_importance',
    'compute_shap_values',
    'compute_learning_curve',
    'compute_confusion_analysis',
    'analyze_hyperparameter_sensitivity',
    'analyze_misclassified_patterns',
    'plot_permutation_importance',
    'plot_shap_summary',
    'plot_shap_dependence',
    'plot_learning_curve',
    'plot_confusion_matrix',
    'plot_sensitivity_curve',
    'plot_misclassification_diff',
]
