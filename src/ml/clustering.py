import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_clustering(
    df: pd.DataFrame,
    max_clusters: int = 6,
):

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
                "At least 20 rows are recommended."
            ),
        }

    numeric_df = numeric_df.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X = imputer.fit_transform(
        numeric_df
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    max_k = min(
        max_clusters,
        len(X) - 1,
    )

    results = []

    for k in range(2, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
        )

        results.append({
            "clusters": k,
            "silhouette_score": round(
                score,
                4,
            ),
        })

    results_df = pd.DataFrame(
        results
    )

    best_k = int(
        results_df.loc[
            results_df[
                "silhouette_score"
            ].idxmax(),
            "clusters",
        ]
    )

    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )

    final_labels = final_model.fit_predict(X)

    clustered_df = df.copy()

    clustered_df["cluster"] = (
        final_labels
    )

    return {
        "available": True,
        "best_clusters": best_k,
        "results": results_df,
        "data": clustered_df,
    }