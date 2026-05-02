"""
Baseline model comparison for the customer-decision-prediction project.

This script trains three baseline classifiers (Logistic Regression, k-NN, Naive Bayes)
using stratified 5-fold cross-validation on three pre-processed datasets, with proper
handling of feature scaling and encoding to avoid data leakage.

Usage: python src/evaluation.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths (script is executed from src/ directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "src" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASETS = [
    {
        'name': 'telco',
        'file': DATA_PROCESSED_DIR / 'telco_clean.csv',
        'target': 'Churn',
        'drop_cols': ['customerID'],  # Non-predictive identifier
        'split_mode': 'random'
    },
    {
        'name': 'bank',
        'file': DATA_PROCESSED_DIR / 'bank_clean.csv',
        'target': 'y',
        'drop_cols': [],
        'split_mode': 'temporal'
    },
    {
        'name': 'ecom',
        'file': DATA_PROCESSED_DIR / 'ecom_clean.csv',
        'target': 'Revenue',
        'drop_cols': [],
        'split_mode': 'random'
    }
]


def load_eda_summary(dataset_name: str) -> dict:
    """Load EDA summary JSON for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'telco', 'bank', 'ecom')
    
    Returns:
        Dictionary containing the EDA summary.
    
    Raises:
        SystemExit: If the summary file does not exist.
    """
    summary_path = REPORTS_DIR / f"summary_{dataset_name}.json"
    if not summary_path.exists():
        logger.error(f"EDA summary not found: {summary_path}")
        sys.exit(1)
    
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON summary {summary_path}: {e}")
        sys.exit(1)


def sort_dataframe_temporally(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame chronologically by month and day_of_week.
    
    Args:
        df: DataFrame with 'month' and 'day_of_week' columns as strings.
    
    Returns:
        Sorted DataFrame with temporal ordering.
    
    Raises:
        KeyError: If 'month' or 'day_of_week' columns are missing.
    """
    month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                   'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    day_order = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
    
    # Konvertiere Strings in Kleinbuchstaben und mappe auf Zahlen
    month_num = df['month'].str.lower().map(month_order)
    day_num = df['day_of_week'].str.lower().map(day_order)
    
    # Kombinierten Sortierschlüssel bilden (Monat * 100 + Tag)
    df['_time_idx'] = month_num * 100 + day_num
    df = df.sort_values('_time_idx').reset_index(drop=True)
    df.drop(columns=['_time_idx'], inplace=True)
    return df


def validate_data_against_summary(X: pd.DataFrame, eda_summary: dict, dataset_name: str, target_col: str, drop_cols: list):
    """Validate feature matrix against EDA summary (including column names).
    
    Args:
        X: Feature matrix (after dropping target and non-predictive columns).
        eda_summary: EDA summary dictionary.
        dataset_name: Name of the dataset.
        target_col: Name of the target column.
        drop_cols: Columns that were dropped (e.g., customerID).
    
    Raises:
        SystemExit: If validation fails.
    """
    expected_rows, expected_cols_raw = eda_summary['shape']
    
    # Extract all column names from EDA summary dtypes
    if 'dtypes' not in eda_summary:
        logger.error(f"{dataset_name}: EDA summary lacks 'dtypes' key.")
        sys.exit(1)
    
    all_cols_in_summary = set(eda_summary['dtypes'].keys())
    
    # Expected feature columns: all columns minus target and drop_cols
    expected_feature_cols = all_cols_in_summary - {target_col} - set(drop_cols)
    actual_feature_cols = set(X.columns)
    
    # Check row count
    if X.shape[0] != expected_rows:
        logger.warning(
            f"{dataset_name}: Row count mismatch (expected {expected_rows}, got {X.shape[0]}). "
            "This is expected if cleaning removed duplicates. Continuing."
        )
    
    # Check feature column names
    if expected_feature_cols != actual_feature_cols:
        missing_cols = expected_feature_cols - actual_feature_cols
        extra_cols = actual_feature_cols - expected_feature_cols
        msg = f"{dataset_name}: Column mismatch with EDA summary."
        if missing_cols:
            msg += f"\n  Missing: {sorted(missing_cols)}"
        if extra_cols:
            msg += f"\n  Extra: {sorted(extra_cols)}"
        logger.error(msg)
        sys.exit(1)
    
    logger.info(f"{dataset_name}: Data validation passed. Shape: {X.shape}")


def extract_feature_types_from_summary(eda_summary: dict, feature_columns: list, dataset_name: str):
    """Extract numeric and categorical features from EDA summary dtypes.
    
    Args:
        eda_summary: EDA summary dictionary with 'dtypes' key.
        feature_columns: List of feature column names.
        dataset_name: Name of the dataset (for logging).
    
    Returns:
        Tuple of (numeric_features, categorical_features) or (None, None) if summary lacks dtype info.
    """
    if 'dtypes' not in eda_summary:
        logger.warning(f"{dataset_name}: EDA summary lacks 'dtypes' key; falling back to select_dtypes.")
        return None, None
    
    dtypes_map = eda_summary['dtypes']
    numeric_features = []
    categorical_features = []
    
    for col in feature_columns:
        if col not in dtypes_map:
            logger.warning(f"{dataset_name}: Column '{col}' not in EDA summary dtypes.")
            continue
        
        dtype_str = str(dtypes_map[col]).lower()
        # Numeric: int*, float*
        if dtype_str.startswith('int') or dtype_str.startswith('float'):
            numeric_features.append(col)
        # Categorical: str, object, bool, category
        elif dtype_str in ['str', 'object', 'bool', 'boolean', 'category']:
            categorical_features.append(col)
        else:
            logger.warning(f"{dataset_name}: Unknown dtype '{dtype_str}' for column '{col}'; treating as categorical.")
            categorical_features.append(col)
    
    return numeric_features, categorical_features


def load_and_prepare_data(dataset_config: dict, eda_summary: dict, split_mode: str):
    """Load, validate, and prepare data for modeling.
    
    Args:
        dataset_config: Configuration dict with 'file', 'target', 'drop_cols'.
        eda_summary: EDA summary from JSON.
        split_mode: Split mode ('random' or 'temporal').
    
    Returns:
        Tuple of (X, y, numeric_features, categorical_features).
    
    Raises:
        SystemExit: If data validation or preparation fails.
    """
    dataset_name = dataset_config['name']
    file_path = dataset_config['file']
    target_col = dataset_config['target']
    drop_cols = dataset_config['drop_cols']
    
    # Load CSV
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {dataset_name}: shape {df.shape}")
    
    # Apply temporal sorting if needed
    if split_mode == 'temporal':
        df = sort_dataframe_temporally(df)
        logger.info(f"{dataset_name}: Data sorted temporally by month and day_of_week.")
    
    # Check target column exists
    if target_col not in df.columns:
        logger.error(f"{dataset_name}: Target column '{target_col}' not found.")
        sys.exit(1)
    
    # Drop non-predictive columns
    if drop_cols:
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        logger.info(f"{dataset_name}: Dropped columns {cols_to_drop}")
    
    # Separate X and y
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Validate target: must be binary {0, 1}
    unique_vals = sorted(y.unique())
    if unique_vals != [0, 1]:
        logger.error(
            f"{dataset_name}: Target column contains {set(unique_vals)}, expected {{0, 1}}."
        )
        sys.exit(1)
    
    # Check for missing values in X
    if X.isna().any().any():
        missing_cols = X.columns[X.isna().any()].tolist()
        missing_counts = X[missing_cols].isna().sum().to_dict()
        logger.error(
            f"{dataset_name}: Found unexpected missing values: {missing_counts}"
        )
        sys.exit(1)
    
    # Validate against EDA summary (now with X as feature matrix)
    validate_data_against_summary(X, eda_summary, dataset_name, target_col, drop_cols)
    
    # Extract feature types from EDA summary
    numeric_features, categorical_features = extract_feature_types_from_summary(
        eda_summary, X.columns.tolist(), dataset_name
    )
    
    # Fallback to select_dtypes if EDA summary lacks dtype info
    if numeric_features is None or categorical_features is None:
        numeric_features = X.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()
        logger.info(f"{dataset_name}: Used select_dtypes as fallback for feature type detection.")
    
    # Log class balance
    class_counts = y.value_counts().sort_index()
    class_pcts = (class_counts / len(y) * 100).round(1)
    logger.info(
        f"{dataset_name}: Class distribution: 0={class_pcts.get(0, 0):.1f}%, 1={class_pcts.get(1, 0):.1f}%"
    )
    if class_pcts.min() < 20:
        logger.warning(
            f"{dataset_name}: Detected class imbalance (minority class: {class_pcts.min():.1f}%). "
            "Consider using class_weight or resampling."
        )
    
    logger.info(
        f"{dataset_name}: {len(numeric_features)} numeric, {len(categorical_features)} categorical features."
    )
    
    return X, y, numeric_features, categorical_features


def build_pipelines(numeric_features: list, categorical_features: list):
    """Build three classifier pipelines with appropriate preprocessing.
    
    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
    
    Returns:
        Dictionary mapping model names to Pipeline objects.
    """
    pipelines = {}
    
    # Preprocessing for Logistic Regression and k-NN (with scaling)
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    # Preprocessing for Naive Bayes (no scaling for numeric features)
    preprocessor_nb = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    pipelines['Logistic Regression'] = Pipeline([
        ('preprocessor', preprocessor_scaled),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipelines['k-NN (k=5)'] = Pipeline([
        ('preprocessor', preprocessor_scaled),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])
    
    pipelines['Naive Bayes'] = Pipeline([
        ('preprocessor', preprocessor_nb),
        ('classifier', GaussianNB())
    ])
    
    return pipelines


def evaluate_dataset(X, y, numeric_features, categorical_features, dataset_name, split_mode):
    """Run stratified cross-validation and test-set evaluation for a dataset.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        dataset_name: Name of the dataset.
        split_mode: Split mode ('random' or 'temporal').
    
    Returns:
        Dictionary mapping model names to results (CV and test metrics).
    
    Raises:
        SystemExit: If evaluation fails.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATING: {dataset_name.upper()} (split_mode={split_mode})")
    logger.info(f"{'='*70}")
    
    # Perform train-test split based on split_mode
    if split_mode == 'temporal':
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        logger.info(f"{dataset_name}: Temporal split at index {split_idx}.")
    else:
        # Stratified random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    
    logger.info(
        f"{dataset_name}: Training set {X_train.shape[0]} rows, "
        f"Test set {X_test.shape[0]} rows"
    )
    
    # Build pipelines
    pipelines = build_pipelines(numeric_features, categorical_features)
    
    results_cv = {}
    results_test = {}
    
    # Select cross-validation strategy based on split_mode
    if split_mode == 'temporal':
        cv = TimeSeriesSplit(n_splits=5)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, pipeline in pipelines.items():
        logger.info(f"\n  {model_name}:")
        
        # Cross-validation on training set
        try:
            cv_results = cross_validate(
                pipeline,
                X_train, y_train,
                cv=cv,
                scoring=['accuracy', 'f1', 'roc_auc'],
                n_jobs=-1,
                return_train_score=False
            )
            
            # Extract mean and std as floats
            results_cv[model_name] = {
                'accuracy_mean': float(cv_results['test_accuracy'].mean()),
                'accuracy_std': float(cv_results['test_accuracy'].std()),
                'f1_mean': float(cv_results['test_f1'].mean()),
                'f1_std': float(cv_results['test_f1'].std()),
                'roc_auc_mean': float(cv_results['test_roc_auc'].mean()),
                'roc_auc_std': float(cv_results['test_roc_auc'].std()),
            }
            
            logger.info(
                f"    CV Results (5-fold, mean ± std):"
                f"\n      Accuracy: {results_cv[model_name]['accuracy_mean']:.4f} ± {results_cv[model_name]['accuracy_std']:.4f}"
                f"\n      F1-Score: {results_cv[model_name]['f1_mean']:.4f} ± {results_cv[model_name]['f1_std']:.4f}"
                f"\n      ROC-AUC:  {results_cv[model_name]['roc_auc_mean']:.4f} ± {results_cv[model_name]['roc_auc_std']:.4f}"
            )
        except Exception as e:
            logger.warning(f"    {model_name}: Cross-validation failed: {e}. Setting CV metrics to NaN.")
            results_cv[model_name] = {
                'accuracy_mean': np.nan,
                'accuracy_std': np.nan,
                'f1_mean': np.nan,
                'f1_std': np.nan,
                'roc_auc_mean': np.nan,
                'roc_auc_std': np.nan,
            }
        
        # Fit on full training set and evaluate on test set
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            test_accuracy = float(accuracy_score(y_test, y_pred))
            test_f1 = float(f1_score(y_test, y_pred))
            
            # Compute ROC-AUC if predict_proba is available
            test_roc_auc = np.nan
            try:
                if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    test_roc_auc = float(roc_auc_score(y_test, y_pred_proba))
                else:
                    logger.warning(f"    {model_name}: predict_proba not available, ROC-AUC set to NaN")
            except ValueError as auc_err:
                logger.warning(f"    {model_name}: Could not compute ROC-AUC: {auc_err}")
            
            results_test[model_name] = {
                'accuracy': test_accuracy,
                'f1': test_f1,
                'roc_auc': test_roc_auc
            }
            
            logger.info(
                f"    Test Results (20% hold-out):"
                f"\n      Accuracy: {test_accuracy:.4f}"
                f"\n      F1-Score: {test_f1:.4f}"
                f"\n      ROC-AUC:  {test_roc_auc:.4f}"
            )
        except Exception as e:
            logger.warning(f"    {model_name}: Test evaluation failed: {e}. Setting test metrics to NaN.")
            results_test[model_name] = {
                'accuracy': np.nan,
                'f1': np.nan,
                'roc_auc': np.nan
            }
    
    return {'cv': results_cv, 'test': results_test}


def compile_results(all_results: dict):
    """Compile results into DataFrames and save to CSV.
    
    Args:
        all_results: Dictionary mapping dataset names to result dicts.
    """
    # CV Results DataFrame (numeric columns for reproducibility)
    cv_data = []
    for dataset_name, results in all_results.items():
        for model_name, metrics in results['cv'].items():
            row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy_mean': round(metrics['accuracy_mean'], 4),
                'Accuracy_std': round(metrics['accuracy_std'], 4),
                'F1_mean': round(metrics['f1_mean'], 4),
                'F1_std': round(metrics['f1_std'], 4),
                'ROC_AUC_mean': round(metrics['roc_auc_mean'], 4),
                'ROC_AUC_std': round(metrics['roc_auc_std'], 4),
            }
            cv_data.append(row)
    
    df_cv = pd.DataFrame(cv_data)
    cv_path = PROJECT_ROOT / 'data' / 'results' / 'baseline_cv_results.csv'
    cv_path.parent.mkdir(parents=True, exist_ok=True)   # <-- diese Zeile neu
    df_cv.to_csv(cv_path, index=False)
    logger.info(f"\nCross-validation results saved to {cv_path}")
    print("\n" + "="*90)
    print("CROSS-VALIDATION RESULTS (5-Fold Stratified, Training Set)")
    print("="*90)
    # Format for display
    df_cv_display = df_cv.copy()
    for col in df_cv_display.columns:
        if col not in ['Dataset', 'Model']:
            df_cv_display[col] = df_cv_display[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else 'NaN')
    print(df_cv_display.to_string(index=False))
    
    # Test Results DataFrame (numeric columns for reproducibility)
    test_data = []
    for dataset_name, results in all_results.items():
        for model_name, metrics in results['test'].items():
            row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': round(metrics['accuracy'], 4),
                'F1': round(metrics['f1'], 4),
                'ROC_AUC': round(metrics['roc_auc'], 4),
            }
            test_data.append(row)
    
    df_test = pd.DataFrame(test_data)
    test_path = REPORTS_DIR / 'baseline_test_results.csv'
    df_test.to_csv(test_path, index=False)
    logger.info(f"Test set results saved to {test_path}")
    print("\n" + "="*90)
    print("TEST SET RESULTS (20% Hold-Out, Fitted on Full Training Set)")
    print("="*90)
    # Format for display
    df_test_display = df_test.copy()
    for col in df_test_display.columns:
        if col not in ['Dataset', 'Model']:
            df_test_display[col] = df_test_display[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else 'NaN')
    print(df_test_display.to_string(index=False))
    print("="*90 + "\n")


def main():
    """Main entry point."""
    logger.info("Starting baseline model evaluation...\n")
    
    all_results = {}
    
    for dataset_config in DATASETS:
        dataset_name = dataset_config['name']
        split_mode = dataset_config['split_mode']
        
        # Load EDA summary
        eda_summary = load_eda_summary(dataset_name)
        
        # Load and prepare data
        X, y, numeric_features, categorical_features = load_and_prepare_data(
            dataset_config, eda_summary, split_mode
        )
        
        # Evaluate dataset
        results = evaluate_dataset(X, y, numeric_features, categorical_features, dataset_name, split_mode)
        all_results[dataset_name] = results
    
    # Compile and save results
    compile_results(all_results)
    
    logger.info("✓ Baseline evaluation complete!")


if __name__ == "__main__":
    main()
