"""Exploratory Data Analysis (EDA) for project datasets.

This script analyzes three datasets (Telco, Bank, E-Commerce) and writes
text reports, JSON summaries and correlation heatmaps into reports/.

Usage: python src/analysis.py
"""

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


REPORT_DIR = Path(__file__).parent.parent / "reports"
PLOTS_DIR = REPORT_DIR / "plots"
REPORT_FILE = REPORT_DIR / "eda_report.txt"


def ensure_report_dirs() -> None:
    """Create report directories if they do not exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str, fh=None):
    """Print a message and optionally write it to a file handle.

    Args:
        msg: Message to emit.
        fh: Optional file handle opened for append.
    """
    print(msg)
    if fh is not None:
        fh.write(msg + "\n")


def load_dataset(path: Path, sep: str, id_col: str = None) -> pd.DataFrame:
    """Load a CSV with a given separator, strip column names and remove id column.

    Returns:
        DataFrame with stripped column names and optional ID column removed.
    """
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip()
    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])
    return df


def normalize_text_and_numeric_columns(df: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """Strip text columns, convert blank strings to NaN, and promote numeric-like columns.

    Args:
        df: Input DataFrame.
        threshold: Maximum allowed NaN rate after numeric coercion for conversion.

    Returns:
        Normalized DataFrame.
    """
    normalized = df.copy()
    text_columns = normalized.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        series = normalized[column].apply(lambda value: value.strip() if isinstance(value, str) else value)
        series = series.replace({"": np.nan})

        coerced = pd.to_numeric(series, errors="coerce")
        if len(series) > 0 and coerced.isna().mean() < threshold:
            normalized[column] = coerced
        else:
            normalized[column] = series

    return normalized


def encode_target(series: pd.Series, fh=None) -> pd.Series:
    """Convert a binary target series to 0/1 values.

    Args:
        series: Target series to encode.
        fh: Optional report file handle.

    Returns:
        Numeric series that can be used for correlation analysis.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        encoded = pd.to_numeric(series, errors="coerce")
        if encoded.isna().any():
            mode = encoded.dropna().mode()
            fill_value = float(mode.iloc[0]) if not mode.empty else 0.0
            encoded = encoded.fillna(fill_value)
        return encoded

    truthy = {"yes", "y", "true", "t", "1"}
    falsy = {"no", "n", "false", "f", "0"}

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(lambda value: 1 if value in truthy else 0 if value in falsy else np.nan)

    unknown_mask = normalized.notna() & mapped.isna()
    if unknown_mask.any():
        unknown_values = sorted(normalized[unknown_mask].dropna().unique().tolist())
        log(f"Warning: target encoding encountered unmapped values: {unknown_values}", fh)
        fallback = pd.to_numeric(series, errors="coerce")
        if fallback.isna().any():
            mode = fallback.dropna().mode()
            fill_value = float(mode.iloc[0]) if not mode.empty else 0.0
            fallback = fallback.fillna(fill_value)
        return fallback

    if mapped.isna().any():
        mode = mapped.dropna().mode()
        fill_value = float(mode.iloc[0]) if not mode.empty else 0.0
        mapped = mapped.fillna(fill_value)

    return mapped.astype(float)


def analyze_missing(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing values and common placeholder patterns.

    Returns a dict with per-column missing counts and placeholder suspects.
    """
    total = len(df)
    missing = df.isna().sum()
    missing_pct = (missing / total * 100).round(3)
    cols_with_missing = {c: {"missing": int(missing[c]), "pct": float(missing_pct[c])}
                         for c in df.columns if missing[c] > 0}

    # placeholder suspects in object columns
    placeholders = ["unknown", "nonexistent", "?", "", " "]
    suspects: Dict[str, Dict[str, int]] = {}
    for c in df.select_dtypes(include=["object", "string"]).columns:
        value_counts = df[c].astype(str).str.strip().str.lower().value_counts()
        found = {}
        for p in placeholders:
            if p in value_counts.index:
                found[p] = int(value_counts.loc[p])
        # also check literal '999' as a placeholder candidate
        if '999' in value_counts.index:
            found['999'] = int(value_counts.loc['999'])
        if found:
            suspects[c] = found

    numeric_placeholder_suspects: Dict[str, Dict[str, int]] = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        count_999 = int((df[c] == 999).sum())
        if count_999 > 0:
            numeric_placeholder_suspects[c] = {"999": count_999}

    merged_suspects = {**suspects, **numeric_placeholder_suspects}
    return {"missing_columns": cols_with_missing, "placeholder_suspects": merged_suspects, "total_missing": int(missing.sum())}


def summarize_target(y: pd.Series) -> Dict[str, Any]:
    """Return absolute and percentage distribution of the target series."""
    counts = y.value_counts(dropna=False)
    pct = (counts / counts.sum() * 100).round(3)
    dist = {str(k): {"count": int(v), "pct": float(pct.loc[k])} for k, v in counts.items()}
    return {"distribution": dist}


def analyze_numerical(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Compute statistics, outliers and scale issues for numerical columns."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    if target_col in num_df.columns:
        num_df = num_df.drop(columns=[target_col])

    if num_df.empty:
        return {"description": {}, "constants": [], "outliers": {}, "scale_warnings": []}

    desc = num_df.describe().T
    desc['range'] = (desc['max'] - desc['min']).astype(float)
    desc['skew'] = num_df.skew()
    desc['kurtosis'] = num_df.kurtosis()

    constants = [c for c in num_df.columns if num_df[c].nunique(dropna=False) <= 1]

    outliers: Dict[str, Dict[str, Any]] = {}
    scale_warnings: List[str] = []
    for c in num_df.columns:
        series = num_df[c].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        count_out = int(((series < low) | (series > high)).sum())
        if count_out > 0:
            outliers[c] = {"count": count_out, "low": float(low), "high": float(high)}

        # scale checks: compare max/min and IQR ratios
        if series.size > 0 and series.min() != 0:
            if series.max() / max(series.min(), 1e-12) > 1e2:
                scale_warnings.append(c)
        if iqr > 0:
            max_over_iqr = series.max() / iqr if iqr != 0 else np.inf
            if max_over_iqr > 1e2:
                scale_warnings.append(c)

    return {"description": desc.fillna(0).to_dict(), "constants": constants, "outliers": outliers, "scale_warnings": list(set(scale_warnings))}


def analyze_categorical(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Analyze categorical columns: cardinality, top values, rare categories, casing issues."""
    cat_df = df.select_dtypes(include=["object", "string", "category", "bool"]).copy()
    if target_col in cat_df.columns:
        cat_df = cat_df.drop(columns=[target_col])

    results: Dict[str, Any] = {}
    for c in cat_df.columns:
        vc = cat_df[c].astype(str).str.strip()
        counts = vc.value_counts(dropna=False)
        total = counts.sum()
        top10 = counts.head(10).to_dict()
        top10_pct = {k: float(round((v / total) * 100, 3)) for k, v in top10.items()}
        rare = [k for k, v in counts.items() if (v / total) < 0.01]
        cardinality = int(vc.nunique(dropna=False))
        casing_issues = None
        # detect inconsistent casing
        lowered = vc.str.lower()
        if lowered.nunique() < vc.nunique():
            groups = vc.groupby(lowered).unique().to_dict()
            casing_issues = {k: list(map(str, v)) for k, v in groups.items() if len(v) > 1}

        results[c] = {
            "cardinality": cardinality,
            "top10_counts": {k: int(v) for k, v in top10.items()},
            "top10_pct": top10_pct,
            "rare_values": rare,
            "high_cardinality": cardinality > 20,
            "casing_issues": casing_issues,
        }

    return results


def analyze_correlations(df: pd.DataFrame, target: pd.Series, dataset_name: str, report_fh=None) -> Dict[str, Any]:
    """Compute correlation matrix, save heatmap and return top correlations with target."""
    try:
        num_df = df.select_dtypes(include=[np.number]).copy()
        corr_info: Dict[str, Any] = {}
        if num_df.shape[1] >= 1:
            corr = num_df.corr(method='pearson')

            try:
                plt.figure(figsize=(max(8, corr.shape[0]), max(6, corr.shape[1] * 0.5)))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
                plt.title(f'Correlation heatmap: {dataset_name}')
                outpath = PLOTS_DIR / f'corr_heatmap_{dataset_name}.png'
                plt.tight_layout()
                plt.savefig(outpath)
            except Exception as exc:
                log(f"Warning: could not save heatmap for {dataset_name}: {exc}", report_fh)
            finally:
                plt.close()

            if target is not None:
                encoded_target = encode_target(target, report_fh)
                merged = num_df.copy()
                merged['_target_numeric_'] = encoded_target
                corr_with_target = merged.corr(numeric_only=True)['_target_numeric_'].drop('_target_numeric_')
                corr_with_target = corr_with_target.sort_values()

                top_pos = corr_with_target.tail(5).to_dict()
                top_neg = corr_with_target.head(5).to_dict()
                corr_info['top_positive_with_target'] = {k: float(v) for k, v in top_pos.items()}
                corr_info['top_negative_with_target'] = {k: float(v) for k, v in top_neg.items()}

            strong_pairs = []
            for i, c1 in enumerate(corr.columns):
                for j, c2 in enumerate(corr.columns):
                    if j <= i:
                        continue
                    val = corr.iloc[i, j]
                    if abs(val) > 0.9:
                        strong_pairs.append({"pair": (c1, c2), "r": float(val)})
            corr_info['strong_pairs'] = strong_pairs
        else:
            corr_info['message'] = 'Not enough numeric columns to compute correlations.'

        return corr_info
    except Exception as exc:
        log(f"Warning: correlation analysis failed for {dataset_name}: {exc}", report_fh)
        return {"error": str(exc)}


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Return number and percent of fully duplicated rows."""
    total = len(df)
    dup_count = int(df.duplicated().sum())
    pct = float(round(dup_count / max(total, 1) * 100, 3))
    return {"duplicates": dup_count, "duplicates_pct": pct}


def analyze_quality(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Run a set of data quality checks and return findings."""
    findings: Dict[str, Any] = {}
    # leading/trailing spaces in object columns
    lt_spaces: Dict[str, int] = {}
    for c in df.select_dtypes(include=["object", "string"]).columns:
        series = df[c].astype(str)
        mism = (series != series.str.strip()).sum()
        if mism > 0:
            lt_spaces[c] = int(mism)
    findings['leading_trailing_spaces'] = lt_spaces

    # numeric columns containing strings that coerce to NaN
    coerced: Dict[str, int] = {}
    for c in df.columns:
        if c == target_col:
            continue
        coerced_series = pd.to_numeric(df[c], errors='coerce')
        bad_count = int((coerced_series.isna() & df[c].notna()).sum())
        if bad_count > 0:
            coerced[c] = bad_count
    findings['numeric_coercion_issues'] = coerced

    # potential id columns
    possible_ids: List[str] = []
    for c in df.columns:
        if df[c].nunique(dropna=False) == len(df):
            possible_ids.append(c)
    findings['possible_id_columns'] = possible_ids

    return findings


def analyze_dataset(config: Dict[str, Any], report_fh) -> Dict[str, Any]:
    """Run full analysis for one dataset and emit report lines and JSON summary.

    Returns a summary dictionary for JSON serialization.
    """
    name = config['name']
    path = Path(config['file'])
    sep = config.get('sep', ',')
    target = config['target']
    id_col = config.get('id_col')

    log(f"\n=== ANALYSIS: {name} ===", report_fh)
    summary: Dict[str, Any] = {"name": name, "file": str(path), "date": datetime.now(timezone.utc).isoformat()}

    df = load_dataset(path, sep=sep, id_col=id_col)
    df = normalize_text_and_numeric_columns(df)

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {name}")

    y = df[target].copy()

    # basic structure
    shape = df.shape
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    mem = int(df.memory_usage(deep=True).sum())
    log(f"Shape: {shape}", report_fh)
    log(f"Dtypes: {dtypes}", report_fh)
    log(f"Memory usage (bytes): {mem}", report_fh)
    summary.update({"shape": shape, "dtypes": dtypes, "memory_bytes": mem})

    # missing and placeholders
    miss = analyze_missing(df)
    log(f"Missing summary: {miss['missing_columns']}", report_fh)
    log(f"Placeholder suspects: {miss['placeholder_suspects']}", report_fh)
    summary['missing'] = miss

    # target distribution
    targ = summarize_target(y)
    log(f"Target distribution: {targ['distribution']}", report_fh)
    dist_vals = [v['pct'] for v in targ['distribution'].values()]
    if min(dist_vals) < 20.0:
        log("Warning: A class has less than 20% representation.", report_fh)
        summary['imbalance_warning'] = True
    else:
        summary['imbalance_warning'] = False
    summary['target'] = targ

    # numerical
    num_info = analyze_numerical(df, target)
    log(f"Numerical constants: {num_info['constants']}", report_fh)
    log(f"Numerical outliers (sample): {list(num_info['outliers'].items())[:5]}", report_fh)
    log(f"Scale warnings: {num_info['scale_warnings']}", report_fh)
    summary['numerical'] = {"constants": num_info['constants'], "outliers": num_info['outliers'], "scale_warnings": num_info['scale_warnings']}

    # categorical
    cat_info = analyze_categorical(df, target)
    # record high-cardinality columns
    high_card = [c for c, v in cat_info.items() if v.get('high_cardinality')]
    log(f"Categorical high-cardinality: {high_card}", report_fh)
    summary['categorical'] = {"high_cardinality": high_card}

    # correlations and heatmap
    corr_info = analyze_correlations(df, y, name, report_fh)
    log(
        f"Correlation info: top pos/neg keys: {list(corr_info.get('top_positive_with_target', {}).keys())} / {list(corr_info.get('top_negative_with_target', {}).keys())}",
        report_fh,
    )
    summary['correlations'] = corr_info

    # duplicates
    dup_info = analyze_duplicates(df)
    log(f"Duplicates: {dup_info}", report_fh)
    summary['duplicates'] = dup_info

    # quality checks
    quality = analyze_quality(df, target)
    log(f"Quality issues: {quality}", report_fh)
    summary['quality'] = quality

    # count outlier rows across numerical columns
    outlier_rows = set()
    for col, info in num_info['outliers'].items():
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        idxs = df.index[(series < low) | (series > high)].tolist()
        outlier_rows.update(idxs)
    summary['outlier_rows_count'] = len(outlier_rows)

    # write JSON summary
    summary_path = REPORT_DIR / f"summary_{name}.json"
    with open(summary_path, 'w', encoding='utf8') as jf:
        json.dump(summary, jf, indent=2, default=lambda o: str(o))

    log(f"Saved JSON summary to {summary_path}", report_fh)

    return summary


def main():
    ensure_report_dirs()

    # configuration for datasets
    configs = [
        {"name": "telco", "file": Path(__file__).parent.parent / "data" / "raw" / "telco.csv", "sep": ",", "target": "Churn", "id_col": "customerID"},
        {"name": "bank", "file": Path(__file__).parent.parent / "data" / "raw" / "bank.csv", "sep": ";", "target": "y", "id_col": None},
        {"name": "ecom", "file": Path(__file__).parent.parent / "data" / "raw" / "ecom.csv", "sep": ",", "target": "Revenue", "id_col": None},
    ]

    summary_overall: Dict[str, Any] = {}

    with open(REPORT_FILE, 'a', encoding='utf8') as report_fh:
        header = f"EDA run at {datetime.now(timezone.utc).isoformat()}"
        log(header, report_fh)
        for conf in configs:
            try:
                log(f"Processing dataset: {conf['name']}", report_fh)
                summary = analyze_dataset(conf, report_fh)
                summary_overall[conf['name']] = summary
            except Exception as e:
                log(f"Error processing {conf['name']}: {e}", report_fh)
                continue

        # final recommendations (simple heuristic rules)
        recs: Dict[str, Any] = {}
        for name, summ in summary_overall.items():
            r = []
            if summ.get('imbalance_warning'):
                r.append('Target is imbalanced; consider class weighting or resampling')
            if summ.get('numerical', {}).get('scale_warnings'):
                r.append('Scale numerical features (StandardScaler/RobustScaler)')
            if summ.get('missing', {}).get('total_missing', 0) > 0 or summ.get('missing', {}).get('placeholder_suspects'):
                r.append('Imputation required; investigate placeholders and missingness')
            if summ.get('categorical', {}).get('high_cardinality'):
                r.append('High cardinality categorical features; consider target/ frequency encoding')
            if summ.get('duplicates', {}).get('duplicates', 0) > 0:
                r.append('Duplicates present; consider removing duplicates')
            recs[name] = r

        log("\n=== SUMMARY RECOMMENDATIONS ===", report_fh)
        for name, r in recs.items():
            log(f"{name}: {r}", report_fh)


if __name__ == "__main__":
    main()
