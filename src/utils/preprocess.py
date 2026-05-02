#!/usr/bin/env python3
"""Uniform data cleaning helpers for the Customer-Decision-Prediction project."""

import pandas as pd
import numpy as np
import re


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Drop exact duplicate rows."""
    return df.drop_duplicates()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Normalise column names.
    - Strip leading/trailing whitespace.
    - Replace spaces, slashes and consecutive underscores with a single underscore.
    """
    new_cols = []
    for col in df.columns:
        col = col.strip()
        col = re.sub(r'[\s/]+', '_', col)      # any whitespace or slash -> underscore
        col = re.sub(r'_+', '_', col)          # collapse multiple underscores
        new_cols.append(col)
    df.columns = new_cols
    return df


def strip_string_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Strip leading/trailing whitespace from every string cell."""
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip()
    return df


def replace_placeholders(df: pd.DataFrame, extra_999_cols: list) -> pd.DataFrame:
    """
    Step 4: Replace common placeholder values with NaN.
    - Generic placeholders ('unknown', 'nonexistent', '?', ' ', '') are replaced in all columns.
    - 999 (int/float/string) is replaced ONLY in the columns listed in extra_999_cols
      (to avoid destroying valid measurements).
    """
    generic = ['unknown', 'nonexistent', '?', ' ', '']
    df.replace(generic, np.nan, inplace=True)

    for col in extra_999_cols:
        if col in df.columns:
            df[col] = df[col].replace([999, 999.0, '999', '999.0'], np.nan)
    return df


def retype_numeric_columns(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Step 5: Convert object columns that are mostly numeric to a numeric dtype.
    A column is converted if at least 95% of its non-NaN values can be coerced to numbers.
    Columns in exclude_cols (e.g., the target) are left untouched.
    """
    if exclude_cols is None:
        exclude_cols = []
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col in exclude_cols:
            continue
        converted = pd.to_numeric(df[col], errors='coerce')
        non_nan_orig = df[col].notna()
        if non_nan_orig.sum() == 0:
            df[col] = converted
            continue
        successful = non_nan_orig & converted.notna()
        ratio = successful.sum() / non_nan_orig.sum()
        if ratio >= 0.95:
            df[col] = converted
    return df


def basic_imputation(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Step 6: Basic imputation of missing values.
    - Numeric columns: fill with median.
    - Categorical columns: fill with mode (most frequent value).
    The target column is never imputed.
    """
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)          # no inplace; assign back
            else:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals[0])  # no inplace; assign back
    return df


def encode_target(df: pd.DataFrame, target_col: str, target_map: dict = None) -> pd.DataFrame:
    """
    Step 7: Encode the target variable as binary integer (0/1).
    If target_map is provided, it is used directly (e.g., {'Yes': 1, 'No': 0}).
    Otherwise, unique values are sorted and mapped to 0/1.
    """
    s = df[target_col].astype(str).str.strip()
    if target_map is None:
        unique_vals = sorted(s.unique())
        target_map = {val: i for i, val in enumerate(unique_vals)}
    df[target_col] = s.map(target_map).astype(int)
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    target_map: dict = None,
    extra_999_cols: list = None
) -> pd.DataFrame:
    """
    Run the full 7-step preprocessing pipeline on a DataFrame.
    Returns the cleaned DataFrame.
    """
    if extra_999_cols is None:
        extra_999_cols = []
    df = remove_duplicates(df)
    df = normalize_column_names(df)
    df = strip_string_cells(df)
    df = replace_placeholders(df, extra_999_cols)
    df = retype_numeric_columns(df, exclude_cols=[target_col])
    df = basic_imputation(df, target_col)
    df = encode_target(df, target_col, target_map)
    return df
