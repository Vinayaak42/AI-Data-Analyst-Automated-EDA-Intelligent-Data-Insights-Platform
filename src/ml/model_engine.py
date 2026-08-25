import pandas as pd

from .target_detector import (
    detect_best_target,
)

from .classification import (
    run_classification,
)

from .regression import (
    run_regression,
)

from .clustering import (
    run_clustering,
)


def determine_problem_type(
    df: pd.DataFrame,
    target: str,
):

    y = df[target].dropna()

    if pd.api.types.is_numeric_dtype(y):

        unique_ratio = (
            y.nunique() / len(y)
            if len(y) > 0
            else 0
        )

        # Numerical values with few unique
        # values are often classification labels.

        if (
            y.nunique() <= 10
            and unique_ratio < 0.1
        ):
            return "classification"

        return "regression"

    return "classification"


def generate_ml_analysis(
    df: pd.DataFrame,
):

    target, candidates = (
        detect_best_target(df)
    )

    result = {
        "target": target,
        "target_candidates": candidates,
        "problem_type": None,
        "supervised": None,
        "clustering": None,
    }

    # --------------------------------
    # SUPERVISED ML
    # --------------------------------

    if target is not None:

        problem_type = determine_problem_type(
            df,
            target,
        )

        result["problem_type"] = (
            problem_type
        )

        if problem_type == "classification":

            result["supervised"] = (
                run_classification(
                    df,
                    target,
                )
            )

        else:

            result["supervised"] = (
                run_regression(
                    df,
                    target,
                )
            )

    # --------------------------------
    # UNSUPERVISED ML
    # --------------------------------

    result["clustering"] = (
        run_clustering(df)
    )

    return result