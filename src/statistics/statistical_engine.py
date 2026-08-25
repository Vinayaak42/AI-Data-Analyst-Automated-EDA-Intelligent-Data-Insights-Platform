import pandas as pd
import numpy as np


class StatisticalEngine:

    def __init__(self, df):

        self.df = df

    # =========================================================
    # DESCRIPTIVE STATISTICS
    # =========================================================

    def descriptive_statistics(self):

        numerical = self.df.select_dtypes(
            include=np.number
        )

        if numerical.empty:
            return {}

        result = {}

        for column in numerical.columns:

            series = numerical[column].dropna()

            if series.empty:
                continue

            result[column] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "q1": round(float(series.quantile(0.25)), 4),
                "q3": round(float(series.quantile(0.75)), 4),
                "skewness": round(float(series.skew()), 4),
            }

        return result

    # =========================================================
    # OUTLIER SUMMARY
    # =========================================================

    def outlier_summary(self):

        numerical = self.df.select_dtypes(
            include=np.number
        )

        results = []

        for column in numerical.columns:

            series = numerical[column].dropna()

            if len(series) < 10:
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
                | (series > upper)
            )

            count = int(mask.sum())

            percentage = (
                count / len(series)
            ) * 100

            results.append({
                "column": column,
                "outlier_count": count,
                "outlier_percentage": round(
                    percentage,
                    2
                )
            })

        return results

    # =========================================================
    # CARDINALITY
    # =========================================================

    def cardinality_analysis(self):

        results = []

        for column in self.df.columns:

            unique = self.df[column].nunique(
                dropna=True
            )

            percentage = (
                unique / len(self.df)
            ) * 100 if len(self.df) else 0

            results.append({
                "column": column,
                "unique_values": int(unique),
                "unique_percentage": round(
                    percentage,
                    2
                )
            })

        return results

    # =========================================================
    # MASTER ANALYSIS
    # =========================================================

    def analyze(self):

        return {
            "descriptive_statistics":
                self.descriptive_statistics(),

            "outliers":
                self.outlier_summary(),

            "cardinality":
                self.cardinality_analysis(),
        }