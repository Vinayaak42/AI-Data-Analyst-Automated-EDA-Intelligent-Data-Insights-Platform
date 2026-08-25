import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02,
):
    """
    Detect anomalous rows using Isolation Forest.
    """

    numeric_df = df.select_dtypes(
        include="number"
    ).copy()

    if numeric_df.shape[1] < 2:
        return {
            "available": False,
            "message": (
                "At least two numerical "
                "columns are required."
            ),
        }

    if len(numeric_df) < 20:
        return {
            "available": False,
            "message": (
                "At least 20 rows are "
                "recommended."
            ),
        }

    # Replace infinite values
    numeric_df = numeric_df.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    # Impute missing values
    imputer = SimpleImputer(
        strategy="median"
    )

    X = imputer.fit_transform(
        numeric_df
    )

    # Scale
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )

    predictions = model.fit_predict(
        X_scaled
    )

    anomaly_scores = model.decision_function(
        X_scaled
    )

    result = df.copy()

    result["_anomaly"] = (
        predictions == -1
    )

    result["_anomaly_score"] = (
        anomaly_scores
    )

    anomaly_count = int(
        result["_anomaly"].sum()
    )

    anomaly_percentage = (
        anomaly_count / len(result) * 100
    )

    anomalies = result[
        result["_anomaly"]
    ].copy()

    return {
        "available": True,
        "total_rows": len(result),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round(
            anomaly_percentage,
            2,
        ),
        "anomalies": anomalies,
    }