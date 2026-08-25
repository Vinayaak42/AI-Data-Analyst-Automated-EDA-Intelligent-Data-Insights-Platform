# ============================================================
# ai_engine.py
# RELIABLE AI INSIGHTS ENGINE
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# SAFE VALUE FORMATTER
# ============================================================

def fmt(value):

    if value is None:
        return "N/A"

    if isinstance(value, (np.integer,)):
        return f"{int(value):,}"

    if isinstance(value, (np.floating, float)):

        if np.isnan(value) or np.isinf(value):
            return "N/A"

        return f"{float(value):,.2f}"

    return str(value)


# ============================================================
# LOAD LOCAL LLM
# ============================================================

def load_local_llm():

    try:

        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM
        )

        model_name = "google/flan-t5-base"

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        model.to(device)
        model.eval()

        return {
            "tokenizer": tokenizer,
            "model": model,
            "device": device
        }

    except Exception as e:

        print("Local LLM unavailable:", e)

        return None


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(
    llm,
    prompt,
    max_tokens=300
):

    if llm is None:
        return None

    try:

        import torch

        tokenizer = llm["tokenizer"]
        model = llm["model"]
        device = llm["device"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=2
            )

        return tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

    except Exception as e:

        print("LLM generation error:", e)

        return None


# ============================================================
# AUTOMATIC INSIGHTS
# ============================================================

def generate_deterministic_insights(df):

    insights = []

    rows = len(df)
    cols = len(df.columns)

    # --------------------------------------------------------
    # DATASET SIZE
    # --------------------------------------------------------

    insights.append(
        f"📊 The dataset contains {rows:,} rows "
        f"and {cols:,} columns."
    )

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    insights.append(
        f"🔢 {len(numeric)} numeric columns and "
        f"{len(categorical)} categorical columns "
        f"were detected."
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing = df.isna().sum()

    total_missing = int(
        missing.sum()
    )

    if total_missing > 0:

        worst = (
            missing
            .sort_values(
                ascending=False
            )
            .head(5)
        )

        insights.append(
            f"⚠️ The dataset contains "
            f"{total_missing:,} missing cells."
        )

        for column, count in worst.items():

            if count > 0:

                pct = (
                    count /
                    max(rows, 1) *
                    100
                )

                insights.append(
                    f"• {column}: "
                    f"{int(count):,} missing "
                    f"({pct:.1f}%)."
                )

    else:

        insights.append(
            "✅ No missing values were detected."
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicates = int(
        df.duplicated().sum()
    )

    if duplicates > 0:

        insights.append(
            f"♻️ {duplicates:,} duplicate rows "
            f"were detected."
        )

    else:

        insights.append(
            "✅ No duplicate rows were detected."
        )

    # --------------------------------------------------------
    # NUMERIC ANALYSIS
    # --------------------------------------------------------

    for col in numeric:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) < 5:
            continue

        mean = series.mean()
        median = series.median()
        minimum = series.min()
        maximum = series.max()

        # Skew

        skew = series.skew()

        if skew > 1:

            distribution = "strongly right-skewed"

        elif skew > 0.5:

            distribution = "right-skewed"

        elif skew < -1:

            distribution = "strongly left-skewed"

        elif skew < -0.5:

            distribution = "left-skewed"

        else:

            distribution = "approximately symmetric"

        insights.append(
            f"📈 **{col}** has mean "
            f"{fmt(mean)}, median "
            f"{fmt(median)}, minimum "
            f"{fmt(minimum)}, maximum "
            f"{fmt(maximum)} and is "
            f"{distribution}."
        )

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    for col in numeric:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) < 5:
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
            |
            (series > upper)
        ).sum()

        percentage = (
            outliers /
            len(series) *
            100
        )

        if percentage >= 5:

            insights.append(
                f"🚨 **{col}** contains "
                f"{int(outliers):,} potential "
                f"outliers ({percentage:.1f}%)."
            )

    # --------------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------------

    if len(numeric) >= 2:

        corr = df[numeric].corr()

        pairs = []

        for i in range(len(numeric)):

            for j in range(i + 1, len(numeric)):

                value = corr.iloc[i, j]

                if pd.isna(value):
                    continue

                pairs.append(
                    (
                        abs(value),
                        numeric[i],
                        numeric[j],
                        value
                    )
                )

        pairs.sort(
            reverse=True
        )

        for _, col1, col2, value in pairs[:5]:

            if abs(value) >= 0.6:

                direction = (
                    "positive"
                    if value > 0
                    else "negative"
                )

                insights.append(
                    f"🔗 **{col1}** and "
                    f"**{col2}** show a strong "
                    f"{direction} relationship "
                    f"(correlation = {value:.2f})."
                )

    # --------------------------------------------------------
    # CATEGORICAL DOMINANCE
    # --------------------------------------------------------

    for col in categorical:

        counts = (
            df[col]
            .value_counts(
                dropna=False
            )
        )

        if len(counts) == 0:
            continue

        top_percentage = (
            counts.iloc[0] /
            rows *
            100
        )

        if top_percentage >= 60:

            insights.append(
                f"📌 **{col}** is highly "
                f"concentrated: the most common "
                f"value represents "
                f"{top_percentage:.1f}% of rows."
            )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    if total_missing > 0:

        insights.append(
            "💡 Recommendation: investigate "
            "missing-value patterns before "
            "building machine-learning models."
        )

    if duplicates > 0:

        insights.append(
            "💡 Recommendation: verify whether "
            "duplicate rows are legitimate "
            "records or accidental duplicates."
        )

    if not numeric:

        insights.append(
            "💡 Recommendation: encode relevant "
            "categorical variables if machine "
            "learning will be performed."
        )

    return insights


# ============================================================
# AI EXECUTIVE SUMMARY
# ============================================================

def generate_ai_summary(
    df,
    llm=None
):

    insights = generate_deterministic_insights(
        df
    )

    # Always return useful insights
    # even if LLM fails.

    if llm is None:

        return insights

    try:

        numeric = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        prompt = f"""
You are a senior data analyst.

Create a short executive summary from
the VERIFIED observations below.

Dataset size:
{len(df)} rows
{len(df.columns)} columns

Numeric columns:
{numeric}

Verified observations:
{" ".join(insights[:15])}

Write:

1. Executive Summary
2. Key Findings
3. Risks
4. Recommendations

Do not invent numbers.
Use only the observations supplied.
Keep the response concise.
"""

        ai_text = generate_text(
            llm,
            prompt,
            max_tokens=350
        )

        if ai_text and len(ai_text.strip()) > 30:

            return [
                ai_text
            ]

    except Exception as e:

        print(
            "AI summary failed:",
            e
        )

    return insights


# ============================================================
# FULL AI ANALYSIS
# ============================================================

def generate_ai_insights(
    df,
    llm=None
):

    insights = generate_deterministic_insights(
        df
    )

    summary = generate_ai_summary(
        df,
        llm
    )

    return {

        "insights":
            insights,

        "summary":
            summary
    }