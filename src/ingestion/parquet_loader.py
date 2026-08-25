import pandas as pd


def load_parquet(file):
    """
    Load Parquet file into a pandas DataFrame.
    """

    return pd.read_parquet(file)