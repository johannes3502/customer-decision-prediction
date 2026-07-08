import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.preprocess import (
    remove_duplicates,
    normalize_column_names,
    strip_string_cells,
    replace_placeholders,
    retype_numeric_columns,
    basic_imputation,
    encode_target,
    preprocess_data,
)


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = remove_duplicates(df)
        assert len(result) == 3


class TestNormalizeColumnNames:
    def test_strips_whitespace(self):
        df = pd.DataFrame({" col1 ": [1], "col2": [2]})
        result = normalize_column_names(df)
        assert list(result.columns) == ["col1", "col2"]

    def test_replaces_spaces(self):
        df = pd.DataFrame({"my column": [1]})
        result = normalize_column_names(df)
        assert list(result.columns) == ["my_column"]


class TestStripStringCells:
    def test_strips_whitespace(self):
        df = pd.DataFrame({"a": [" foo ", "bar"]})
        result = strip_string_cells(df)
        assert result["a"].tolist() == ["foo", "bar"]


class TestReplacePlaceholders:
    def test_replaces_unknown(self):
        df = pd.DataFrame({"a": ["unknown", "foo", "?"]})
        result = replace_placeholders(df, extra_999_cols=[])
        assert result["a"].isna().sum() == 2
        assert result["a"].iloc[1] == "foo"

    def test_replaces_999_in_specified_cols(self):
        df = pd.DataFrame({"a": [999, 1, 2], "b": [999, 1, 2]})
        result = replace_placeholders(df, extra_999_cols=["a"])
        assert result["a"].isna().sum() == 1
        assert result["b"].iloc[0] == 999


class TestRetypeNumericColumns:
    def test_converts_numeric_strings(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = retype_numeric_columns(df)
        assert pd.api.types.is_numeric_dtype(result["a"])


class TestBasicImputation:
    def test_imputes_numeric_median(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "target": [0, 1, 0]})
        result = basic_imputation(df, "target")
        assert result["a"].iloc[1] == 2.0

    def test_imputes_categorical_mode(self):
        df = pd.DataFrame({"a": ["x", np.nan, "x"], "target": [0, 1, 0]})
        result = basic_imputation(df, "target")
        assert result["a"].iloc[1] == "x"


class TestEncodeTarget:
    def test_encodes_with_map(self):
        df = pd.DataFrame({"target": ["Yes", "No", "Yes"]})
        result = encode_target(df, "target", {"Yes": 1, "No": 0})
        assert result["target"].tolist() == [1, 0, 1]

    def test_encodes_auto_map(self):
        df = pd.DataFrame({"target": ["A", "B", "A"]})
        result = encode_target(df, "target")
        assert result["target"].tolist() == [0, 1, 0]


class TestPreprocessData:
    def test_full_pipeline(self):
        df = pd.DataFrame({
            "col a": ["1.0", "2.0", "3.0"],
            "col_b": [" x ", " y ", "unknown"],
            "target": ["Yes", "No", "Yes"],
        })
        result = preprocess_data(
            df, "target",
            target_map={"Yes": 1, "No": 0},
            extra_999_cols=[],
        )
        assert list(result.columns) == ["col_a", "col_b", "target"]
        assert result["target"].tolist() == [1, 0, 1]
        assert pd.api.types.is_numeric_dtype(result["col_a"])
        assert result["col_b"].isna().sum() == 0
        assert set(result["col_b"].dropna().unique()) == {"x", "y"}
