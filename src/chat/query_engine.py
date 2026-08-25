import duckdb
import pandas as pd


class QueryEngine:

    def __init__(self, df: pd.DataFrame):

        self.df = df

        self.connection = duckdb.connect(
            database=":memory:"
        )

        self.connection.register(
            "dataset",
            self.df
        )

    def execute(self, sql: str):

        result = self.connection.execute(
            sql
        ).df()

        return result

    def schema(self):

        return pd.DataFrame({
            "column": self.df.columns,
            "dtype": [
                str(dtype)
                for dtype in self.df.dtypes
            ],
        })

    def close(self):

        self.connection.close()