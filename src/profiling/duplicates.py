import pandas as pd


def analyze_duplicates(df: pd.DataFrame) -> dict:
    """
    Analyze duplicate records.
    """

    duplicate_count = int(df.duplicated().sum())

    duplicate_percentage = (
        duplicate_count / len(df) * 100
        if len(df) > 0
        else 0
    )

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(
            duplicate_percentage,
            2
        ),
        "has_duplicates": duplicate_count > 0,
    }