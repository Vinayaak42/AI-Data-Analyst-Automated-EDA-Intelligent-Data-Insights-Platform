import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .preprocessor import build_preprocessor


def run_regression(
    df: pd.DataFrame,
    target: str,
):

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    mask = y.notna()

    X = X.loc[mask]
    y = y.loc[mask]

    if not pd.api.types.is_numeric_dtype(y):

        return {
            "available": False,
            "message": (
                "Regression requires a "
                "numerical target."
            ),
        }

    if len(y) < 30:

        return {
            "available": False,
            "message": (
                "At least 30 rows are recommended."
            ),
        }

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    preprocessor = build_preprocessor(
        X_train
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
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

            mse = mean_squared_error(
                y_test,
                predictions,
            )

            results.append({

                "model": name,

                "mae": round(
                    mean_absolute_error(
                        y_test,
                        predictions,
                    ),
                    4,
                ),

                "rmse": round(
                    mse ** 0.5,
                    4,
                ),

                "r2": round(
                    r2_score(
                        y_test,
                        predictions,
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

    best_model = None

    if "r2" in result_df.columns:

        valid = result_df[
            result_df["r2"].notna()
        ]

        if not valid.empty:

            best_model = valid.sort_values(
                "r2",
                ascending=False,
            ).iloc[0]["model"]

    return {
        "available": True,
        "problem_type": "regression",
        "target": target,
        "results": result_df,
        "best_model": best_model,
    }