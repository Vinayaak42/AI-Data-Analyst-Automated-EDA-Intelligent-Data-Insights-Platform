import pandas as pd


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values for every column.
    """

    result = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum().values,
        "total_rows": len(df),
    })

    result["missing_percentage"] = (
        result["missing_count"] / result["total_rows"] * 100
    ).round(2)

    result["status"] = result["missing_percentage"].apply(
        lambda x:
            "Critical" if x >= 30
            else "High" if x >= 10
            else "Moderate" if x > 0
            else "Good"
    )

    return result.sort_values(
        "missing_percentage",
        ascending=False
    ).reset_index(drop=True)