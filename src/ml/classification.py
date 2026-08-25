import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .preprocessor import build_preprocessor


def run_classification(
    df: pd.DataFrame,
    target: str,
):

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # Remove rows with missing target
    mask = y.notna()

    X = X.loc[mask]
    y = y.loc[mask]

    # Prevent extremely high-cardinality
    # target variables from being treated
    # as normal classification targets.

    if y.nunique() < 2:

        return {
            "available": False,
            "message": "Target has fewer than 2 classes."
        }

    if y.nunique() > 20:

        return {
            "available": False,
            "message": (
                "Target has too many classes "
                "for automatic classification."
            )
        }

    # Limit extremely small classes
    class_counts = y.value_counts()

    if class_counts.min() < 2:

        return {
            "available": False,
            "message": (
                "At least one class has fewer "
                "than 2 samples."
            )
        }

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    preprocessor = build_preprocessor(
        X_train
    )

    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),
    }

    results = []

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        try:

            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = pipeline.predict(
                X_test
            )

            results.append({

                "model": name,

                "accuracy": round(
                    accuracy_score(
                        y_test,
                        predictions,
                    ),
                    4,
                ),

                "precision": round(
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    4,
                ),

                "recall": round(
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    4,
                ),

                "f1": round(
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    4,
                ),

            })

        except Exception as e:

            results.append({
                "model": name,
                "error": str(e),
            })

    result_df = pd.DataFrame(results)

    if "f1" in result_df.columns:

        valid = result_df[
            result_df["f1"].notna()
        ]

        if not valid.empty:

            best_model = valid.sort_values(
                "f1",
                ascending=False,
            ).iloc[0]["model"]

        else:

            best_model = None

    else:

        best_model = None

    return {
        "available": True,
        "problem_type": "classification",
        "target": target,
        "results": result_df,
        "best_model": best_model,
    }