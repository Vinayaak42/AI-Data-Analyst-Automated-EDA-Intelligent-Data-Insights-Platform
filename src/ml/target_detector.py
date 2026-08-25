import pandas as pd


TARGET_KEYWORDS = [
    "target",
    "label",
    "outcome",
    "class",
    "response",
    "churn",
    "price",
    "sales",
    "revenue",
    "profit",
    "default",
    "fraud",
    "status",
]


def detect_target_candidates(df: pd.DataFrame) -> list:
    """
    Detect possible target columns using
    column names and dataset characteristics.
    """

    candidates = []

    for column in df.columns:

        name = column.lower().strip()

        score = 0
        reasons = []

        # --------------------------------
        # Keyword detection
        # --------------------------------

        for keyword in TARGET_KEYWORDS:

            if keyword in name:

                score += 3

                reasons.append(
                    f"Name contains '{keyword}'"
                )

                break

        # --------------------------------
        # Missing value check
        # --------------------------------

        missing_ratio = (
            df[column].isna().mean()
        )

        if missing_ratio < 0.1:

            score += 1

        # --------------------------------
        # ID-like columns
        # --------------------------------

        unique_ratio = (
            df[column].nunique(dropna=False)
            / len(df)
            if len(df) > 0
            else 0
        )

        if unique_ratio > 0.98:

            score -= 3

            reasons.append(
                "Likely identifier"
            )

        # --------------------------------
        # Classification-like target
        # --------------------------------

        unique_count = (
            df[column].nunique()
        )

        if 2 <= unique_count <= 20:

            score += 2

            reasons.append(
                "Low/moderate cardinality"
            )

        # --------------------------------
        # Add candidate
        # --------------------------------

        if score > 0:

            candidates.append({
                "column": column,
                "score": score,
                "reasons": reasons,
                "unique_values": unique_count,
            })

    return sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )


def detect_best_target(
    df: pd.DataFrame,
):
    candidates = detect_target_candidates(df)

    if not candidates:
        return None, []

    return (
        candidates[0]["column"],
        candidates,
    )