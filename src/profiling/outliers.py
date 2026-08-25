import pandas as pd


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect numerical outliers using the IQR method.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    results = []

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        percentage = (
            outlier_count / len(series) * 100
        )

        results.append({
            "column": column,
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "lower_bound": round(lower_bound, 3),
            "upper_bound": round(upper_bound, 3),
            "outlier_count": outlier_count,
            "outlier_percentage": round(
                percentage,
                2
            ),
        })

    return pd.DataFrame(results).sort_values(
        "outlier_percentage",
        ascending=False
    ).reset_index(drop=True)