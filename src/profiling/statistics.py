import pandas as pd


def numerical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate statistics for numerical columns.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    stats = numeric_df.describe().T

    stats["missing"] = numeric_df.isnull().sum()

    stats["missing_percentage"] = (
        numeric_df.isnull().mean() * 100
    ).round(2)

    stats["unique_values"] = (
        numeric_df.nunique()
    )

    stats["skewness"] = numeric_df.skew()

    stats["kurtosis"] = numeric_df.kurtosis()

    return stats.reset_index().rename(
        columns={"index": "column"}
    )


def categorical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate statistics for categorical columns.
    """

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    if categorical_df.empty:
        return pd.DataFrame()

    rows = []

    for column in categorical_df.columns:

        rows.append({
            "column": column,
            "unique_values": df[column].nunique(),
            "missing_values": df[column].isnull().sum(),
            "top_value": (
                df[column].mode().iloc[0]
                if not df[column].mode().empty
                else None
            ),
            "top_value_count": (
                df[column].value_counts().iloc[0]
                if not df[column].value_counts().empty
                else 0
            ),
        })

    return pd.DataFrame(rows)