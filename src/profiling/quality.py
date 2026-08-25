import pandas as pd


def calculate_quality_score(df: pd.DataFrame) -> dict:
    """
    Calculate an overall data quality score.
    """

    if df.empty:
        return {
            "score": 0,
            "grade": "F"
        }

    total_cells = df.shape[0] * df.shape[1]

    missing_cells = int(
        df.isnull().sum().sum()
    )

    missing_ratio = (
        missing_cells / total_cells
        if total_cells > 0
        else 0
    )

    duplicate_ratio = (
        df.duplicated().sum() / len(df)
        if len(df) > 0
        else 0
    )

    missing_penalty = min(
        missing_ratio * 100,
        40
    )

    duplicate_penalty = min(
        duplicate_ratio * 100,
        20
    )

    score = 100 - missing_penalty - duplicate_penalty

    score = max(
        0,
        min(100, score)
    )

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": round(score, 2),
        "grade": grade,
        "missing_cells": missing_cells,
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
    }