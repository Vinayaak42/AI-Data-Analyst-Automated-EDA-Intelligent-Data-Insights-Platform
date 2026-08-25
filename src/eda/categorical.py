import pandas as pd
import plotly.express as px


def categorical_bar_chart(
    df: pd.DataFrame,
    column: str,
    top_n: int = 15,
):
    """
    Create a bar chart for categorical data.
    """

    counts = (
        df[column]
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    counts.columns = [
        column,
        "count",
    ]

    fig = px.bar(
        counts,
        x=column,
        y="count",
        title=f"Top Categories — {column}",
        text="count",
    )

    fig.update_layout(
        height=450,
        template="plotly_white",
        xaxis_title=column,
        yaxis_title="Count",
    )

    return fig