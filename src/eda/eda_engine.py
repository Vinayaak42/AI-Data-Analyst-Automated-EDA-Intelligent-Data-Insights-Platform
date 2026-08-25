import pandas as pd

from .numerical import (
    numerical_histogram,
    numerical_boxplot,
)

from .categorical import (
    categorical_bar_chart,
)

from .relationships import (
    scatter_plot,
    correlation_heatmap,
)

from .time_series import (
    detect_datetime_columns,
    time_series_plot,
)


def generate_eda(df: pd.DataFrame):
    """
    Automatically determine which EDA
    visualizations should be generated.
    """

    result = {
        "numerical": [],
        "categorical": [],
        "scatter": [],
        "correlation": None,
        "time_series": [],
    }

    # --------------------------------
    # NUMERICAL COLUMNS
    # --------------------------------

    numerical_columns = (
        df.select_dtypes(
            include="number"
        ).columns.tolist()
    )

    for column in numerical_columns:

        result["numerical"].append({
            "column": column,
            "histogram": numerical_histogram(
                df,
                column
            ),
            "boxplot": numerical_boxplot(
                df,
                column
            ),
        })

    # --------------------------------
    # CATEGORICAL COLUMNS
    # --------------------------------

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns.tolist()
    )

    for column in categorical_columns:

        # Don't create charts for extremely
        # high-cardinality columns.

        if df[column].nunique(
            dropna=True
        ) <= 100:

            result["categorical"].append({
                "column": column,
                "chart": categorical_bar_chart(
                    df,
                    column
                ),
            })

    # --------------------------------
    # CORRELATION
    # --------------------------------

    if len(numerical_columns) >= 2:

        result["correlation"] = (
            correlation_heatmap(df)
        )

    # --------------------------------
    # SCATTER PLOTS
    # --------------------------------

    # Limit automatic pair generation
    # to avoid hundreds of charts.

    max_scatter_columns = min(
        len(numerical_columns),
        5
    )

    scatter_columns = (
        numerical_columns[
            :max_scatter_columns
        ]
    )

    for i in range(
        len(scatter_columns)
    ):

        for j in range(
            i + 1,
            len(scatter_columns)
        ):

            x = scatter_columns[i]
            y = scatter_columns[j]

            result["scatter"].append({
                "x": x,
                "y": y,
                "chart": scatter_plot(
                    df,
                    x,
                    y
                ),
            })

    # --------------------------------
    # TIME SERIES
    # --------------------------------

    datetime_columns = (
        detect_datetime_columns(df)
    )

    for date_column in datetime_columns:

        for value_column in numerical_columns:

            chart = time_series_plot(
                df,
                date_column,
                value_column,
            )

            if chart is not None:

                result["time_series"].append({
                    "date": date_column,
                    "value": value_column,
                    "chart": chart,
                })

    return result