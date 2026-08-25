import pandas as pd
import plotly.express as px


def detect_datetime_columns(
    df: pd.DataFrame,
):
    """
    Detect columns that are likely to contain dates.
    """

    datetime_columns = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            datetime_columns.append(column)
            continue

        if df[column].dtype == "object":

            sample = df[column].dropna().head(100)

            if sample.empty:
                continue

            converted = pd.to_datetime(
                sample,
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.8:
                datetime_columns.append(column)

    return datetime_columns


def time_series_plot(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
):
    """
    Create a time-series chart.
    """

    plot_df = df[
        [date_column, value_column]
    ].copy()

    plot_df[date_column] = pd.to_datetime(
        plot_df[date_column],
        errors="coerce"
    )

    plot_df = plot_df.dropna()

    if plot_df.empty:
        return None

    plot_df = (
        plot_df
        .sort_values(date_column)
    )

    fig = px.line(
        plot_df,
        x=date_column,
        y=value_column,
        title=(
            f"{value_column} over time"
        ),
    )

    fig.update_layout(
        height=450,
        template="plotly_white",
    )

    return fig