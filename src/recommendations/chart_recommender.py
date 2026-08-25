import pandas as pd
import numpy as np


class ChartRecommender:

    def __init__(self, df):

        self.df = df

    def recommend(self):

        recommendations = []

        numerical = list(
            self.df.select_dtypes(
                include=np.number
            ).columns
        )

        categorical = list(
            self.df.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool"
                ]
            ).columns
        )

        datetime_columns = list(
            self.df.select_dtypes(
                include=[
                    "datetime",
                    "datetimetz"
                ]
            ).columns
        )

        # ------------------------------------------------------
        # NUMERICAL
        # ------------------------------------------------------

        for column in numerical[:10]:

            recommendations.append({
                "chart": "Histogram",
                "column": column,
                "reason": (
                    "Useful for understanding "
                    "the distribution of the variable."
                )
            })

            recommendations.append({
                "chart": "Box Plot",
                "column": column,
                "reason": (
                    "Useful for detecting "
                    "outliers and spread."
                )
            })

        # ------------------------------------------------------
        # CATEGORICAL
        # ------------------------------------------------------

        for column in categorical[:10]:

            unique_count = (
                self.df[column]
                .nunique(dropna=True)
            )

            if 2 <= unique_count <= 20:

                recommendations.append({
                    "chart": "Bar Chart",
                    "column": column,
                    "reason": (
                        "Useful for comparing "
                        "category frequencies."
                    )
                })

        # ------------------------------------------------------
        # RELATIONSHIPS
        # ------------------------------------------------------

        if len(numerical) >= 2:

            for i in range(
                min(len(numerical), 5)
            ):

                for j in range(
                    i + 1,
                    min(len(numerical), 5)
                ):

                    recommendations.append({
                        "chart": "Scatter Plot",
                        "x": numerical[i],
                        "y": numerical[j],
                        "reason": (
                            "Useful for identifying "
                            "relationships between variables."
                        )
                    })

        # ------------------------------------------------------
        # TIME SERIES
        # ------------------------------------------------------

        if datetime_columns and numerical:

            for date_column in datetime_columns:

                for value_column in numerical[:5]:

                    recommendations.append({
                        "chart": "Line Chart",
                        "x": date_column,
                        "y": value_column,
                        "reason": (
                            "Useful for analyzing "
                            "changes over time."
                        )
                    })

        return recommendations