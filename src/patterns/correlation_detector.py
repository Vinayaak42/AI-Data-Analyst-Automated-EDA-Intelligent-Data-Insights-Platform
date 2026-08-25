import pandas as pd


def detect_correlations(
    df: pd.DataFrame,
    strong_threshold: float = 0.7,
):
    """
    Detect strong positive and negative
    correlations between numerical variables.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return []

    correlation_matrix = numeric_df.corr()

    results = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            col1 = columns[i]
            col2 = columns[j]

            value = correlation_matrix.loc[
                col1,
                col2
            ]

            if abs(value) >= strong_threshold:

                if value > 0:
                    relationship = "Strong Positive"
                else:
                    relationship = "Strong Negative"

                results.append({
                    "column_1": col1,
                    "column_2": col2,
                    "correlation": round(
                        float(value),
                        3
                    ),
                    "relationship": relationship,
                })

    return sorted(
        results,
        key=lambda x: abs(
            x["correlation"]
        ),
        reverse=True,
    )