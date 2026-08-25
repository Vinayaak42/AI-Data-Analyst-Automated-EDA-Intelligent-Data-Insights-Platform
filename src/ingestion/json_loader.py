import pandas as pd


def load_json(file):
    """
    Load JSON file into a pandas DataFrame.
    """

    return pd.read_json(file)