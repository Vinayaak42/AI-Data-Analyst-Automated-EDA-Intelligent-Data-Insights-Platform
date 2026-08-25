import json


class AIReportGenerator:

    def __init__(self, ai_engine):

        self.ai_engine = ai_engine

    def build_context(
        self,
        df,
        profile,
        eda,
        patterns,
        ml_analysis,
        recommendations,
        statistics,
    ):

        context = {

            "dataset": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": list(df.columns),
            },

            "profile": profile,

            "patterns": patterns,

            "ml_analysis": ml_analysis,

            "recommendations": recommendations,

            "statistics": statistics,
        }

        return context

    # =========================================================
    # REPORT GENERATION
    # =========================================================

    def generate_report(
        self,
        df,
        profile,
        eda,
        patterns,
        ml_analysis,
        recommendations,
        statistics,
    ):

        context = self.build_context(
            df=df,
            profile=profile,
            eda=eda,
            patterns=patterns,
            ml_analysis=ml_analysis,
            recommendations=recommendations,
            statistics=statistics,
        )

        prompt = f"""
You are an expert AI Data Analyst.

Analyze the following dataset analysis results.

IMPORTANT RULES:

1. Use ONLY the provided evidence.
2. Never invent statistics.
3. Never invent relationships.
4. Clearly distinguish correlation from causation.
5. Prioritize the most important findings.
6. Explain technical results in simple business language.
7. Give actionable recommendations.
8. Mention important data-quality issues.
9. If there is insufficient evidence, explicitly say so.

Generate a professional data analyst report with these sections:

# Executive Summary

Give 3-5 important findings.

# Dataset Overview

Explain:
- number of rows
- number of columns
- important column types
- data quality

# Key Insights

Give the most important statistical findings.

# Trends

Explain detected trends.

# Relationships

Explain important correlations or relationships.

# Anomalies

Explain detected outliers or anomalies.

# Machine Learning Findings

Explain:
- problem type
- detected target
- best model
- important ML results

# Data Quality Issues

Explain missing values, duplicates, suspicious values
and other issues.

# Recommendations

Give practical actions.

# Priority Actions

Give the top 5 things the user should investigate first.

Here is the analysis evidence:

{json.dumps(context, default=str, indent=2)}
"""

        return self.ai_engine.generate_text(prompt)