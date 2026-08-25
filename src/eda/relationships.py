import pandas as pd
import plotly.express as px


def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
):
    """
    Create a scatter plot between two numerical columns.
    """

    plot_df = df[[x, y]].dropna()

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        title=f"{x} vs {y}",
        opacity=0.65,
    )

    fig.update_layout(
        height=450,
        template="plotly_white",
    )

    return fig


def correlation_heatmap(
    df: pd.DataFrame,
):
    """
    Create correlation heatmap.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return None

    correlation = numeric_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix",
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
    )

    return fig