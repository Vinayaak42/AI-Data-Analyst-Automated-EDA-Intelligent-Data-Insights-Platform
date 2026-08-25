# ============================================================
# ml_engine.py
# AUTOMATIC ML READINESS + FEATURE ANALYSIS
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# ML READINESS
# ============================================================

def ml_readiness(
    df,
    target
):

    if target not in df.columns:

        return {
            "ready": False,
            "reason": "Target column not found."
        }

    y = df[target]

    missing = (
        y.isna().mean() * 100
    )

    unique = (
        y.nunique()
    )

    rows = len(df)

    if missing > 30:

        return {
            "ready": False,
            "reason":
                "Target contains more than 30% missing values."
        }

    if unique <= 1:

        return {
            "ready": False,
            "reason":
                "Target contains only one unique value."
        }

    if rows < 50:

        return {
            "ready": False,
            "reason":
                "Dataset contains fewer than 50 rows."
        }

    if pd.api.types.is_numeric_dtype(y):

        task = (
            "Regression"
            if unique > 20
            else
            "Classification or Regression"
        )

    else:

        task = "Classification"

    return {

        "ready":
            True,

        "task":
            task,

        "target":
            target,

        "rows":
            rows,

        "unique_target_values":
            unique,

        "target_missing_percentage":
            round(
                missing,
                2
            )
    }


# ============================================================
# TARGET CORRELATIONS
# ============================================================

def target_correlations(
    df,
    target
):

    if target not in df.columns:
        return pd.DataFrame()

    if not pd.api.types.is_numeric_dtype(
        df[target]
    ):

        return pd.DataFrame()

    numeric = df.select_dtypes(
        include=np.number
    )

    if target not in numeric.columns:
        return pd.DataFrame()

    corr = (
        numeric
        .corr()[target]
        .drop(target)
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    return (
        corr
        .reset_index()
        .rename(
            columns={
                "index":
                    "Feature",

                target:
                    "Correlation"
            }
        )
    )


# ============================================================
# CLASS BALANCE
# ============================================================

def class_balance(
    df,
    target
):

    if target not in df.columns:
        return pd.DataFrame()

    if pd.api.types.is_numeric_dtype(
        df[target]
    ):

        if df[target].nunique() > 20:
            return pd.DataFrame()

    result = (
        df[target]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    result.columns = [
        "Class",
        "Count"
    ]

    result["Percentage"] = (
        result["Count"]
        /
        result["Count"].sum()
        * 100
    ).round(2)

    return result