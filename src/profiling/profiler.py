import pandas as pd

from .missing import analyze_missing_values
from .duplicates import analyze_duplicates
from .statistics import (
    numerical_statistics,
    categorical_statistics,
)
from .outliers import detect_outliers
from .quality import calculate_quality_score


def detect_column_types(df: pd.DataFrame) -> dict:

    numerical = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime_columns,
    }


def detect_constant_columns(
    df: pd.DataFrame
) -> list:

    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]


def detect_possible_id_columns(
    df: pd.DataFrame
) -> list:

    possible_ids = []

    for column in df.columns:

        unique_ratio = (
            df[column].nunique(dropna=False)
            / len(df)
            if len(df) > 0
            else 0
        )

        column_lower = column.lower()

        if (
            unique_ratio >= 0.95
            or column_lower.endswith("_id")
            or column_lower == "id"
        ):
            possible_ids.append(column)

    return possible_ids


def generate_profile(df: pd.DataFrame) -> dict:

    column_types = detect_column_types(df)

    profile = {

        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },

        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum()
            / 1024**2,
            2
        ),

        "column_types": column_types,

        "missing_values":
            analyze_missing_values(df),

        "duplicates":
            analyze_duplicates(df),

        "numerical_statistics":
            numerical_statistics(df),

        "categorical_statistics":
            categorical_statistics(df),

        "outliers":
            detect_outliers(df),

        "quality":
            calculate_quality_score(df),

        "constant_columns":
            detect_constant_columns(df),

        "possible_id_columns":
            detect_possible_id_columns(df),
    }

    return profile