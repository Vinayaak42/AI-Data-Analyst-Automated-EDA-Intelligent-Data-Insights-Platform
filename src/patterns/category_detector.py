import pandas as pd


def detect_category_patterns(
    df: pd.DataFrame,
    top_n: int = 10,
):
    """
    Detect important patterns between
    categorical and numerical variables.
    """

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
        ]
    ).columns

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    results = []

    for category in categorical_columns:

        # Avoid extremely high-cardinality columns
        if df[category].nunique(
            dropna=True
        ) > 50:
            continue

        for metric in numerical_columns:

            grouped = (
                df.groupby(category)[metric]
                .agg(
                    ["mean", "median", "count"]
                )
                .dropna()
            )

            if grouped.empty:
                continue

            highest = grouped["mean"].idxmax()
            lowest = grouped["mean"].idxmin()

            results.append({
                "category": category,
                "metric": metric,
                "highest_category": str(
                    highest
                ),
                "highest_mean": round(
                    float(
                        grouped.loc[
                            highest,
                            "mean"
                        ]
                    ),
                    2,
                ),
                "lowest_category": str(
                    lowest
                ),
                "lowest_mean": round(
                    float(
                        grouped.loc[
                            lowest,
                            "mean"
                        ]
                    ),
                    2,
                ),
            })

    return results