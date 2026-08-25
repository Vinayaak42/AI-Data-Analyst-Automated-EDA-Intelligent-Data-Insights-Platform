# ============================================================
# data_analyzer.py
# ADVANCED AUTOMATED EDA ENGINE
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# BASIC DATA TYPE DETECTION
# ============================================================

def detect_column_types(df):

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    datetime_cols = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[col]
        ):
            datetime_cols.append(col)

    # Try detecting date-like object columns

    for col in categorical.copy():

        if col in datetime_cols:
            continue

        try:

            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.80:

                datetime_cols.append(col)

                if col in categorical:
                    categorical.remove(col)

        except Exception:
            pass

    return (
        numeric,
        categorical,
        datetime_cols
    )


# ============================================================
# DATA QUALITY SCORE
# ============================================================

def calculate_quality_score(df):

    if df.empty:
        return 0

    total_cells = (
        df.shape[0] *
        df.shape[1]
    )

    if total_cells == 0:
        return 0

    missing_cells = (
        df.isna().sum().sum()
    )

    duplicate_rows = (
        df.duplicated().sum()
    )

    missing_penalty = (
        missing_cells /
        total_cells *
        50
    )

    duplicate_penalty = (
        duplicate_rows /
        max(len(df), 1) *
        30
    )

    constant_columns = sum(
        df[col].nunique(dropna=False) <= 1
        for col in df.columns
    )

    constant_penalty = (
        constant_columns /
        max(len(df.columns), 1) *
        20
    )

    score = (
        100
        - missing_penalty
        - duplicate_penalty
        - constant_penalty
    )

    return round(
        max(0, min(100, score)),
        2
    )


# ============================================================
# NUMERIC SUMMARY
# ============================================================

def numeric_summary(df, numeric_columns):

    result = {}

    for col in numeric_columns:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        result[col] = {

            "count":
                int(series.count()),

            "mean":
                float(series.mean()),

            "median":
                float(series.median()),

            "std":
                float(series.std()),

            "min":
                float(series.min()),

            "max":
                float(series.max()),

            "q1":
                float(series.quantile(0.25)),

            "q3":
                float(series.quantile(0.75)),

            "skew":
                float(series.skew()),

            "unique":
                int(series.nunique()),

            "missing":
                int(df[col].isna().sum())
        }

    return result


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def detect_outliers(
    df,
    numeric_columns
):

    results = []

    for col in numeric_columns:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (series < lower)
            |
            (series > upper)
        )

        count = int(mask.sum())

        percentage = (
            count /
            len(series) *
            100
        )

        results.append({

            "column":
                col,

            "outliers":
                count,

            "percentage":
                round(
                    percentage,
                    2
                ),

            "lower_bound":
                round(
                    lower,
                    3
                ),

            "upper_bound":
                round(
                    upper,
                    3
                )
        })

    results.sort(
        key=lambda x:
        x["percentage"],
        reverse=True
    )

    return results


# ============================================================
# CORRELATIONS
# ============================================================

def correlation_analysis(
    df,
    numeric_columns
):

    results = []

    if len(numeric_columns) < 2:
        return results

    corr = df[
        numeric_columns
    ].corr()

    for i in range(
        len(numeric_columns)
    ):

        for j in range(
            i + 1,
            len(numeric_columns)
        ):

            value = corr.iloc[
                i,
                j
            ]

            if pd.isna(value):
                continue

            results.append({

                "column_1":
                    numeric_columns[i],

                "column_2":
                    numeric_columns[j],

                "correlation":
                    round(
                        float(value),
                        4
                    ),

                "strength":
                    correlation_strength(
                        value
                    )
            })

    results.sort(
        key=lambda x:
        abs(
            x["correlation"]
        ),
        reverse=True
    )

    return results


# ============================================================
# CORRELATION STRENGTH
# ============================================================

def correlation_strength(value):

    value = abs(value)

    if value >= 0.8:
        return "Very Strong"

    if value >= 0.6:
        return "Strong"

    if value >= 0.4:
        return "Moderate"

    if value >= 0.2:
        return "Weak"

    return "Very Weak"


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

def categorical_analysis(
    df,
    categorical_columns
):

    results = {}

    for col in categorical_columns:

        counts = (
            df[col]
            .value_counts(
                dropna=False
            )
            .head(20)
        )

        results[col] = {

            "unique":
                int(
                    df[col].nunique(
                        dropna=True
                    )
                ),

            "top":
                str(
                    counts.index[0]
                )
                if len(counts)
                else None,

            "top_count":
                int(
                    counts.iloc[0]
                )
                if len(counts)
                else 0,

            "distribution":
                {
                    str(k):
                    int(v)
                    for k, v
                    in counts.items()
                }
        }

    return results


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def missing_analysis(df):

    result = []

    for col in df.columns:

        missing = int(
            df[col].isna().sum()
        )

        percentage = (
            missing /
            max(len(df), 1) *
            100
        )

        result.append({

            "column":
                col,

            "missing":
                missing,

            "percentage":
                round(
                    percentage,
                    2
                )
        })

    result.sort(
        key=lambda x:
        x["percentage"],
        reverse=True
    )

    return result


# ============================================================
# CONSTANT COLUMNS
# ============================================================

def constant_columns(df):

    return [
        col
        for col in df.columns
        if df[col].nunique(
            dropna=False
        ) <= 1
    ]


# ============================================================
# HIGH CARDINALITY
# ============================================================

def high_cardinality_columns(df):

    result = []

    for col in df.columns:

        unique = df[col].nunique()

        ratio = (
            unique /
            max(len(df), 1)
        )

        if ratio >= 0.90:

            result.append({

                "column":
                    col,

                "unique":
                    unique,

                "ratio":
                    round(
                        ratio * 100,
                        2
                    )
            })

    return result


# ============================================================
# DISTRIBUTION ANALYSIS
# ============================================================

def distribution_analysis(
    df,
    numeric_columns
):

    result = []

    for col in numeric_columns:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) < 5:
            continue

        skew = series.skew()

        if abs(skew) < 0.5:
            shape = "Approximately symmetric"

        elif skew > 0:
            shape = "Right skewed"

        else:
            shape = "Left skewed"

        result.append({

            "column":
                col,

            "skewness":
                round(
                    float(skew),
                    3
                ),

            "distribution":
                shape
        })

    return result


# ============================================================
# PATTERN DETECTION
# ============================================================

def detect_patterns(
    df,
    numeric_columns,
    categorical_columns
):

    patterns = []

    # --------------------------------------------------------
    # NUMERIC TRENDS
    # --------------------------------------------------------

    for col in numeric_columns:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) < 10:
            continue

        first = series.head(
            max(1, len(series) // 10)
        ).mean()

        last = series.tail(
            max(1, len(series) // 10)
        ).mean()

        if first == 0:
            continue

        change = (
            (last - first)
            /
            abs(first)
            *
            100
        )

        if abs(change) >= 20:

            direction = (
                "increasing"
                if change > 0
                else "decreasing"
            )

            patterns.append(
                f"{col} shows a possible "
                f"{direction} pattern of "
                f"approximately "
                f"{abs(change):.1f}%."
            )

    # --------------------------------------------------------
    # CATEGORICAL DOMINANCE
    # --------------------------------------------------------

    for col in categorical_columns:

        counts = (
            df[col]
            .value_counts(
                normalize=True,
                dropna=False
            )
        )

        if len(counts) == 0:
            continue

        top_ratio = counts.iloc[0]

        if top_ratio >= 0.60:

            patterns.append(
                f"{col} is dominated by "
                f"one category "
                f"({top_ratio * 100:.1f}%)."
            )

    return patterns


# ============================================================
# FULL ANALYSIS CONTEXT
# ============================================================

def build_analysis_context(df):

    numeric_columns, categorical_columns, datetime_columns = (
        detect_column_types(df)
    )

    context = {

        "rows":
            int(len(df)),

        "columns":
            int(len(df.columns)),

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "datetime_columns":
            datetime_columns,

        "missing_cells":
            int(
                df.isna().sum().sum()
            ),

        "missing_percentage":
            float(
                df.isna().mean().mean()
                * 100
            ),

        "duplicate_rows":
            int(
                df.duplicated().sum()
            ),

        "quality_score":
            calculate_quality_score(
                df
            ),

        "numeric_summary":
            numeric_summary(
                df,
                numeric_columns
            ),

        "outliers":
            detect_outliers(
                df,
                numeric_columns
            ),

        "correlations":
            correlation_analysis(
                df,
                numeric_columns
            ),

        "categorical_summary":
            categorical_analysis(
                df,
                categorical_columns
            ),

        "missing_analysis":
            missing_analysis(
                df
            ),

        "constant_columns":
            constant_columns(
                df
            ),

        "high_cardinality":
            high_cardinality_columns(
                df
            ),

        "distributions":
            distribution_analysis(
                df,
                numeric_columns
            ),

        "patterns":
            detect_patterns(
                df,
                numeric_columns,
                categorical_columns
            )
    }

    return context


# ============================================================
# COLUMN PROFILE
# ============================================================

def get_column_profile(df):

    result = []

    for col in df.columns:

        result.append({

            "Column":
                col,

            "Type":
                str(
                    df[col].dtype
                ),

            "Rows":
                len(df),

            "Missing":
                int(
                    df[col].isna().sum()
                ),

            "Missing %":
                round(
                    df[col].isna().mean()
                    * 100,
                    2
                ),

            "Unique":
                int(
                    df[col].nunique()
                ),

            "Unique %":
                round(
                    df[col].nunique()
                    /
                    max(len(df), 1)
                    * 100,
                    2
                )
        })

    return pd.DataFrame(
        result
    )