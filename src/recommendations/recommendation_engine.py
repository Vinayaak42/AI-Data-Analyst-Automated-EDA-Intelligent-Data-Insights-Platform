import pandas as pd
import numpy as np


class RecommendationEngine:

    def __init__(self, df):

        self.df = df

    # ==========================================================
    # DATA QUALITY RECOMMENDATIONS
    # ==========================================================

    def data_quality_recommendations(self):

        recommendations = []

        total_rows = len(self.df)

        if total_rows == 0:
            return recommendations

        # Missing values
        for column in self.df.columns:

            missing_count = self.df[column].isna().sum()

            missing_percentage = (
                missing_count / total_rows
            ) * 100

            if missing_percentage >= 50:

                recommendations.append({
                    "priority": "🔴 High",
                    "category": "Data Quality",
                    "title": f"High missing values in {column}",
                    "message": (
                        f"{column} contains "
                        f"{missing_percentage:.1f}% missing values."
                    ),
                    "recommendation": (
                        "Consider removing the column "
                        "or applying an appropriate "
                        "missing-value strategy."
                    )
                })

            elif missing_percentage >= 10:

                recommendations.append({
                    "priority": "🟠 Medium",
                    "category": "Data Quality",
                    "title": f"Missing values in {column}",
                    "message": (
                        f"{column} contains "
                        f"{missing_percentage:.1f}% missing values."
                    ),
                    "recommendation": (
                        "Investigate why values are missing "
                        "before performing ML analysis."
                    )
                })

        # Duplicate rows

        duplicates = self.df.duplicated().sum()

        if duplicates > 0:

            percentage = (
                duplicates / total_rows
            ) * 100

            recommendations.append({
                "priority": "🟠 Medium",
                "category": "Data Quality",
                "title": "Duplicate records detected",
                "message": (
                    f"{duplicates:,} duplicate rows "
                    f"({percentage:.1f}%) were detected."
                ),
                "recommendation": (
                    "Review duplicate records before "
                    "building machine-learning models."
                )
            })

        return recommendations

    # ==========================================================
    # CORRELATION RECOMMENDATIONS
    # ==========================================================

    def correlation_recommendations(self):

        recommendations = []

        numerical = self.df.select_dtypes(
            include=np.number
        )

        if numerical.shape[1] < 2:
            return recommendations

        correlation = numerical.corr()

        columns = correlation.columns

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                col1 = columns[i]
                col2 = columns[j]

                value = correlation.loc[
                    col1,
                    col2
                ]

                if pd.isna(value):
                    continue

                strength = abs(value)

                if strength >= 0.8:

                    direction = (
                        "positive"
                        if value > 0
                        else "negative"
                    )

                    recommendations.append({
                        "priority": "🔴 High",
                        "category": "Relationship",
                        "title": (
                            f"Strong relationship: "
                            f"{col1} ↔ {col2}"
                        ),
                        "message": (
                            f"Correlation = {value:.2f}. "
                            f"This is a strong "
                            f"{direction} relationship."
                        ),
                        "recommendation": (
                            "Investigate this relationship "
                            "for business or predictive insights."
                        )
                    })

                elif strength >= 0.6:

                    recommendations.append({
                        "priority": "🟡 Low",
                        "category": "Relationship",
                        "title": (
                            f"Moderate relationship: "
                            f"{col1} ↔ {col2}"
                        ),
                        "message": (
                            f"Correlation = {value:.2f}."
                        ),
                        "recommendation": (
                            "Consider investigating this "
                            "relationship further."
                        )
                    })

        return recommendations

    # ==========================================================
    # NUMERICAL DISTRIBUTION
    # ==========================================================

    def distribution_recommendations(self):

        recommendations = []

        numerical = self.df.select_dtypes(
            include=np.number
        )

        for column in numerical.columns:

            series = numerical[column].dropna()

            if len(series) < 10:
                continue

            skewness = series.skew()

            if abs(skewness) >= 2:

                direction = (
                    "right-skewed"
                    if skewness > 0
                    else "left-skewed"
                )

                recommendations.append({
                    "priority": "🟡 Low",
                    "category": "Distribution",
                    "title": (
                        f"Highly skewed variable: {column}"
                    ),
                    "message": (
                        f"{column} is strongly "
                        f"{direction} "
                        f"(skewness={skewness:.2f})."
                    ),
                    "recommendation": (
                        "Consider transformations or "
                        "robust statistical methods."
                    )
                })

        return recommendations

    # ==========================================================
    # OUTLIER DETECTION
    # ==========================================================

    def outlier_recommendations(self):

        recommendations = []

        numerical = self.df.select_dtypes(
            include=np.number
        )

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

            outliers = (
                (series < lower)
                | (series > upper)
            ).sum()

            percentage = (
                outliers / len(series)
            ) * 100

            if percentage >= 5:

                recommendations.append({
                    "priority": "🟠 Medium",
                    "category": "Anomaly",
                    "title": (
                        f"Potential outliers in {column}"
                    ),
                    "message": (
                        f"Approximately "
                        f"{percentage:.1f}% of values "
                        f"appear to be outliers."
                    ),
                    "recommendation": (
                        "Investigate extreme values "
                        "before modeling."
                    )
                })

        return recommendations

    # ==========================================================
    # MASTER ENGINE
    # ==========================================================

    def generate(self):

        recommendations = []

        recommendations.extend(
            self.data_quality_recommendations()
        )

        recommendations.extend(
            self.correlation_recommendations()
        )

        recommendations.extend(
            self.distribution_recommendations()
        )

        recommendations.extend(
            self.outlier_recommendations()
        )

        priority_order = {
            "🔴 High": 1,
            "🟠 Medium": 2,
            "🟡 Low": 3
        }

        recommendations.sort(
            key=lambda x: priority_order.get(
                x["priority"],
                99
            )
        )

        return recommendations