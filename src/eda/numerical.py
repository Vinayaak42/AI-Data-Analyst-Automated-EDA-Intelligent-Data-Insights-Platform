import pandas as pd
import plotly.express as px


def numerical_histogram(df: pd.DataFrame, column: str):
    """
    Create histogram for a numerical column.
    """

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        marginal="box",
        nbins=30,
    )

    fig.update_layout(
        height=450,
        template="plotly_white",
    )

    return fig


def numerical_boxplot(df: pd.DataFrame, column: str):
    """
    Create box plot for a numerical column.
    """

    fig = px.box(
        df,
        y=column,
        title=f"Box Plot — {column}",
        points="outliers",
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
    )

    return fig