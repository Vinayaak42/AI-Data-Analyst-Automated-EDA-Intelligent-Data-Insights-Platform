# src/ai/prompt_builder.py

def build_insight_prompt(
    df,
    profile,
    patterns,
    ml_analysis,
):
    """
    Build a structured prompt for the local AI model.

    The prompt is designed to produce evidence-based
    business insights from the analytical results.
    """

    # ---------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------

    rows, columns = df.shape

    numerical_columns = list(
        df.select_dtypes(include="number").columns
    )

    categorical_columns = list(
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    # ---------------------------------------------------------
    # Convert analysis objects safely to text
    # ---------------------------------------------------------

    profile_text = str(profile)

    patterns_text = str(patterns)

    ml_text = str(ml_analysis)

    # ---------------------------------------------------------
    # Build prompt
    # ---------------------------------------------------------

    prompt = f"""
You are an expert AI Data Analyst.

Analyze the dataset using ONLY the evidence provided below.

Your job is to produce a clear, concise and useful business
analysis report.

============================================================
DATASET OVERVIEW
============================================================

Rows: {rows}
Columns: {columns}

Numerical columns:
{numerical_columns}

Categorical columns:
{categorical_columns}

============================================================
DATASET PROFILE
============================================================

{profile_text}

============================================================
DETECTED PATTERNS
============================================================

{patterns_text}

============================================================
MACHINE LEARNING ANALYSIS
============================================================

{ml_text}

============================================================
REPORT REQUIREMENTS
============================================================

Create an evidence-based AI Data Analyst report.

Use the following structure:

# Executive Summary

Give a concise summary of the most important findings.

# Dataset Overview

Explain the size, structure and important characteristics
of the dataset.

# Data Quality

Discuss:
- Missing values
- Duplicate records
- Potential data quality problems
- Important data-quality observations

# Key Statistical Findings

Explain the most important numerical findings.

# Important Patterns

Discuss:
- Trends
- Correlations
- Category differences
- Anomalies
- Other meaningful relationships

# Machine Learning Findings

Explain:
- Detected target
- Problem type
- Best model if available
- Model performance
- Clustering findings if available

# Business Insights

Translate the technical findings into practical
business-oriented insights.

# Recommendations

Give specific and actionable recommendations.

# Risks and Limitations

Mention important limitations and avoid making claims
that are not supported by the data.

============================================================
IMPORTANT RULES
============================================================

1. Do not invent numbers.
2. Do not invent relationships.
3. Use only information present in the analysis.
4. Clearly distinguish facts from interpretations.
5. Keep the report professional.
6. Prefer concise bullet points where appropriate.
7. Explain technical results in understandable language.
8. Highlight the most important findings first.

Generate the final report now.
"""

    return prompt