import pandas as pd
import numpy as np
from scipy.stats import linregress


def detect_numeric_trends(df: pd.DataFrame) -> list:
    """
    Detect increasing, decreasing, or stable trends
    in numerical columns based on row order.
    """

    results = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 10:
            continue

        x = np.arange(len(series))
        y = series.values

        slope, _, r_value, _, _ = linregress(x, y)

        mean_value = series.mean()

        if mean_value == 0:
            continue

        relative_change = (
            slope * len(series)
        ) / abs(mean_value) * 100

        if abs(r_value) < 0.3:
            direction = "Stable / Weak Trend"

        elif slope > 0:
            direction = "Increasing"

        else:
            direction = "Decreasing"

        results.append({
            "column": column,
            "direction": direction,
            "slope": round(float(slope), 4),
            "r_squared": round(float(r_value ** 2), 4),
            "relative_change_percent": round(
                float(relative_change),
                2
            ),
        })

    return results