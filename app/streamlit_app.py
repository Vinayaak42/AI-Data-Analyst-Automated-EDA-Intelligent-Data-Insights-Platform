# ============================================================
# AI DATA ANALYST
# Python + Statistics + EDA + ML + NLP/LLM
# Data Engineering + Visualization + Streamlit + AI Automation
# ============================================================

import re
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .section-title {
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 15px;
        background: #1e293b;
        border: 1px solid #334155;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

if "ai_context" not in st.session_state:
    st.session_state.ai_context = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# LOCAL LLM
# ============================================================

@st.cache_resource(show_spinner=False)
def load_local_llm():

    try:

        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
        )

    except Exception:

        return None

    model_name = "google/flan-t5-base"

    try:

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
            "device": device,
        }

    except Exception as e:

        print(
            "LLM loading error:",
            e
        )

        return None


def generate_local_text(
    prompt,
    max_tokens=300
):

    llm = load_local_llm()

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
            max_length=2048,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=2,
            )

        text = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        return text.strip()

    except Exception as e:

        print(
            "Generation error:",
            e
        )

        return None


# ============================================================
# DATA LOADING
# ============================================================

def load_uploaded_file(
    uploaded_file
):

    try:

        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):

            return pd.read_csv(
                uploaded_file
            )

        elif filename.endswith(
            (".xlsx", ".xls")
        ):

            return pd.read_excel(
                uploaded_file
            )

        elif filename.endswith(".json"):

            return pd.read_json(
                uploaded_file
            )

        elif filename.endswith(".parquet"):

            return pd.read_parquet(
                uploaded_file
            )

        else:

            st.error(
                "Unsupported file format."
            )

            return None

    except Exception as e:

        st.error(
            f"Could not read file: {e}"
        )

        return None


# ============================================================
# EXECUTE SQL
# ============================================================

def execute_sql(
    df,
    sql
):

    try:

        import duckdb

        conn = duckdb.connect(
            database=":memory:"
        )

        conn.register(
            "dataset",
            df
        )

        result = conn.execute(
            sql
        ).df()

        conn.close()

        return result, None

    except Exception as e:

        return None, str(e)


# ============================================================
# DATASET SCHEMA
# ============================================================

def get_dataset_schema(
    df
):

    schema = []

    for column in df.columns:

        schema.append(
            {
                "column": column,
                "dtype": str(
                    df[column].dtype
                ),
                "unique_values": int(
                    df[column].nunique(
                        dropna=True
                    )
                ),
                "missing": int(
                    df[column].isna().sum()
                ),
                "examples": (
                    df[column]
                    .dropna()
                    .astype(str)
                    .head(3)
                    .tolist()
                ),
            }
        )

    return schema


# ============================================================
# AUTOMATIC EDA
# ============================================================

def calculate_eda(
    df
):

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns.tolist()
    )

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    duplicates = int(
        df.duplicated().sum()
    )

    numeric_summary = pd.DataFrame()

    if numeric_columns:

        numeric_summary = (
            df[numeric_columns]
            .describe()
            .T
        )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing": missing,
        "duplicates": duplicates,
        "numeric_summary": numeric_summary,
    }


# ============================================================
# INSIGHT CONTEXT
# ============================================================

def build_insight_context(
    df
):

    context = {}

    context["rows"] = len(df)

    context["columns"] = len(
        df.columns
    )

    context["missing_cells"] = int(
        df.isna().sum().sum()
    )

    context["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    context["missing_percentage"] = round(
        df.isna().mean().mean() * 100,
        2,
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns.tolist()
    )

    context[
        "numeric_columns"
    ] = numeric_columns

    context[
        "categorical_columns"
    ] = categorical_columns

    # --------------------------------------------------------
    # NUMERIC SUMMARY
    # --------------------------------------------------------

    numeric_summary = {}

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(series) == 0:
            continue

        numeric_summary[column] = {

            "mean": round(
                float(
                    series.mean()
                ),
                3,
            ),

            "median": round(
                float(
                    series.median()
                ),
                3,
            ),

            "min": round(
                float(
                    series.min()
                ),
                3,
            ),

            "max": round(
                float(
                    series.max()
                ),
                3,
            ),

            "std": round(
                float(
                    series.std()
                ),
                3,
            ),

            "missing": int(
                df[column].isna().sum()
            ),

            "unique": int(
                df[column].nunique()
            ),
        }

    context[
        "numeric_summary"
    ] = numeric_summary

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    correlations = []

    if len(numeric_columns) >= 2:

        corr = df[
            numeric_columns
        ].corr()

        for i in range(
            len(corr.columns)
        ):

            for j in range(
                i + 1,
                len(corr.columns),
            ):

                value = corr.iloc[
                    i,
                    j,
                ]

                if pd.isna(value):
                    continue

                if abs(value) >= 0.5:

                    correlations.append(
                        {
                            "column_1":
                                corr.columns[i],

                            "column_2":
                                corr.columns[j],

                            "correlation":
                                round(
                                    float(
                                        value
                                    ),
                                    3,
                                ),
                        }
                    )

    correlations.sort(
        key=lambda x:
        abs(
            x["correlation"]
        ),
        reverse=True,
    )

    context[
        "correlations"
    ] = correlations[:15]

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    outliers = []

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        count = int(
            (
                (series < lower)
                |
                (series > upper)
            ).sum()
        )

        if count > 0:

            outliers.append(
                {
                    "column":
                        column,

                    "count":
                        count,

                    "percentage":
                        round(
                            count /
                            len(series)
                            * 100,
                            2,
                        ),
                }
            )

    context[
        "outliers"
    ] = outliers

    return context


# ============================================================
# RULE BASED INSIGHTS
# ============================================================

def generate_rule_based_insights(
    df,
    context
):

    insights = []

    insights.append(
        f"The dataset contains "
        f"{context['rows']:,} rows and "
        f"{context['columns']:,} columns."
    )

    missing_pct = (
        context[
            "missing_percentage"
        ]
    )

    if missing_pct == 0:

        insights.append(
            "No missing values were detected."
        )

    elif missing_pct < 5:

        insights.append(
            f"Missing data is relatively low "
            f"at {missing_pct}%."
        )

    elif missing_pct < 20:

        insights.append(
            f"The dataset has a moderate "
            f"missing-data issue "
            f"({missing_pct}%)."
        )

    else:

        insights.append(
            f"The dataset has a significant "
            f"missing-data issue "
            f"({missing_pct}%)."
        )

    duplicates = (
        context[
            "duplicate_rows"
        ]
    )

    if duplicates > 0:

        pct = round(
            duplicates /
            max(
                context["rows"],
                1,
            )
            * 100,
            2,
        )

        insights.append(
            f"{duplicates:,} duplicate rows "
            f"were detected "
            f"({pct}%)."
        )

    else:

        insights.append(
            "No duplicate rows were detected."
        )

    for item in context[
        "correlations"
    ][:5]:

        relationship = (
            "positive"
            if item[
                "correlation"
            ] > 0
            else "negative"
        )

        strength = (
            "strong"
            if abs(
                item["correlation"]
            ) >= 0.7
            else "moderate"
        )

        insights.append(
            f"{item['column_1']} and "
            f"{item['column_2']} have a "
            f"{strength} {relationship} "
            f"relationship "
            f"(correlation "
            f"{item['correlation']:.2f})."
        )

    for item in context[
        "outliers"
    ][:5]:

        insights.append(
            f"{item['column']} has "
            f"{item['count']} potential "
            f"outliers "
            f"({item['percentage']}%)."
        )

    return insights


# ============================================================
# AI INSIGHTS
# ============================================================

def generate_ai_insights(
    df
):

    context = build_insight_context(
        df
    )

    rule_insights = (
        generate_rule_based_insights(
            df,
            context,
        )
    )

    prompt = f"""
You are a senior data analyst.

Use ONLY the verified statistics below.

Rows:
{context['rows']}

Columns:
{context['columns']}

Missing cells:
{context['missing_cells']}

Missing percentage:
{context['missing_percentage']}%

Duplicate rows:
{context['duplicate_rows']}

Numeric columns:
{context['numeric_columns']}

Categorical columns:
{context['categorical_columns']}

Numeric summary:
{context['numeric_summary']}

Correlations:
{context['correlations']}

Outliers:
{context['outliers']}

Write a professional analysis.

Sections:

EXECUTIVE SUMMARY

KEY INSIGHTS

DATA QUALITY

RELATIONSHIPS

ANOMALIES

RECOMMENDATIONS

IMPORTANT QUESTIONS

Never invent numbers.

Correlation does not prove causation.
"""

    ai_text = generate_local_text(
        prompt,
        max_tokens=500,
    )

    if not ai_text:

        ai_text = (
            "## Executive Summary\n\n"
            "Automatic analysis completed.\n\n"
            "## Key Insights\n\n"
        )

        for insight in rule_insights:

            ai_text += (
                f"- {insight}\n"
            )

        ai_text += (
            "\n## Recommendations\n\n"
            "- Review missing values.\n"
            "- Investigate outliers.\n"
            "- Examine strong correlations.\n"
            "- Validate findings with domain knowledge.\n"
        )

    return ai_text, context


# ============================================================
# SQL CLEANER
# ============================================================

def clean_generated_sql(
    sql
):

    if not sql:
        return None

    sql = sql.strip()

    sql = re.sub(
        r"```sql",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"```",
        "",
        sql,
    )

    match = re.search(
        r"\b(SELECT|WITH)\b",
        sql,
        flags=re.IGNORECASE,
    )

    if match:

        sql = sql[
            match.start():
        ]

    sql = sql.rstrip(";")

    return sql.strip()


# ============================================================
# SQL VALIDATION
# ============================================================

def validate_generated_sql(
    sql,
    df
):

    if not sql:

        return (
            False,
            "Empty SQL query.",
        )

    sql_lower = sql.lower().strip()

    if not (
        sql_lower.startswith(
            "select"
        )
        or
        sql_lower.startswith(
            "with"
        )
    ):

        return (
            False,
            "Only SELECT/WITH queries are allowed.",
        )

    forbidden = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "replace",
        "merge",
        "attach",
        "detach",
        "install",
        "load",
        "pragma",
        "copy",
        "export",
    ]

    for word in forbidden:

        if re.search(
            rf"\b{word}\b",
            sql_lower,
        ):

            return (
                False,
                f"Forbidden SQL operation: {word}",
            )

    tables = re.findall(
        r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        sql_lower,
    )

    for table in tables:

        if table != "dataset":

            return (
                False,
                f"Unknown table: {table}",
            )

    return True, "Valid"


# ============================================================
# FIND COLUMN
# ============================================================

def find_matching_column(
    text,
    columns
):

    text_lower = text.lower()

    for column in columns:

        if column.lower() in text_lower:

            return column

    normalized_text = re.sub(
        r"[^a-z0-9]",
        "",
        text_lower,
    )

    for column in columns:

        normalized_column = re.sub(
            r"[^a-z0-9]",
            "",
            column.lower(),
        )

        if (
            normalized_column
            and
            normalized_column
            in normalized_text
        ):

            return column

    return None


# ============================================================
# RULE BASED SQL
# ============================================================

def rule_based_sql(
    question,
    df
):

    q = question.lower().strip()

    columns = list(
        df.columns
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    if (
        "how many rows" in q
        or
        "row count" in q
        or
        "number of rows" in q
        or
        "how many records" in q
    ):

        return """
SELECT COUNT(*) AS row_count
FROM dataset
"""

    # --------------------------------------------------------
    # SHOW DATA
    # --------------------------------------------------------

    if (
        "show data" in q
        or
        "show me the data" in q
        or
        "preview" in q
    ):

        return """
SELECT *
FROM dataset
LIMIT 20
"""

    # --------------------------------------------------------
    # AVERAGE
    # --------------------------------------------------------

    if (
        "average" in q
        or
        "mean" in q
    ):

        column = find_matching_column(
            q,
            numeric_columns,
        )

        if column:

            safe_column = (
                '"'
                + str(column).replace(
                    '"',
                    '""',
                )
                + '"'
            )

            return f"""
SELECT
    AVG({safe_column}) AS average_value
FROM dataset
"""

    # --------------------------------------------------------
    # MAX
    # --------------------------------------------------------

    if (
        "highest" in q
        or
        "maximum" in q
        or
        re.search(
            r"\bmax\b",
            q,
        )
    ):

        column = find_matching_column(
            q,
            numeric_columns,
        )

        if column:

            safe_column = (
                '"'
                + str(column).replace(
                    '"',
                    '""',
                )
                + '"'
            )

            return f"""
SELECT
    MAX({safe_column}) AS maximum_value
FROM dataset
"""

    # --------------------------------------------------------
    # MIN
    # --------------------------------------------------------

    if (
        "lowest" in q
        or
        "minimum" in q
        or
        re.search(
            r"\bmin\b",
            q,
        )
    ):

        column = find_matching_column(
            q,
            numeric_columns,
        )

        if column:

            safe_column = (
                '"'
                + str(column).replace(
                    '"',
                    '""',
                )
                + '"'
            )

            return f"""
SELECT
    MIN({safe_column}) AS minimum_value
FROM dataset
"""

    # --------------------------------------------------------
    # UNIQUE VALUES
    # --------------------------------------------------------

    if (
        "unique values" in q
        or
        "distinct values" in q
    ):

        column = find_matching_column(
            q,
            columns,
        )

        if column:

            safe_column = (
                '"'
                + str(column).replace(
                    '"',
                    '""',
                )
                + '"'
            )

            return f"""
SELECT
    {safe_column},
    COUNT(*) AS count
FROM dataset
GROUP BY {safe_column}
ORDER BY count DESC
LIMIT 50
"""

    return None


# ============================================================
# AI SQL GENERATOR
# ============================================================

def generate_sql_from_question(
    question,
    df
):

    rule_sql = rule_based_sql(
        question,
        df,
    )

    if rule_sql:

        return clean_generated_sql(
            rule_sql
        )

    schema = get_dataset_schema(
        df
    )

    schema_text = ""

    for item in schema:

        schema_text += (
            f"Column: {item['column']}\n"
            f"Type: {item['dtype']}\n"
            f"Unique: {item['unique_values']}\n"
            f"Examples: {item['examples']}\n\n"
        )

    prompt = f"""
You are an expert DuckDB SQL generator.

Table name:

dataset

Schema:

{schema_text}

User question:

{question}

Rules:

1. Return ONLY SQL.
2. Use SELECT or WITH.
3. Use only dataset.
4. Never modify data.
5. Never create tables.
6. Never drop tables.
7. Never use INSERT.
8. Never use UPDATE.
9. Never use DELETE.
10. Never use PRAGMA.
11. Never use COPY.
12. Use exact column names.
13. Use double quotes around column names.
14. Do not explain.
15. Do not use markdown.

SQL:
"""

    generated = generate_local_text(
        prompt,
        max_tokens=300,
    )

    return clean_generated_sql(
        generated
    )


# ============================================================
# EXPLAIN QUERY RESULT
# ============================================================

def explain_query_result(
    question,
    result
):

    if result is None:

        return (
            "No result was produced."
        )

    if result.empty:

        return (
            "The query executed successfully, "
            "but no records matched the question."
        )

    if (
        result.shape[0] == 1
        and
        result.shape[1] == 1
    ):

        value = result.iloc[
            0,
            0,
        ]

        return (
            f"### Answer\n\n"
            f"The answer is **{value}**."
        )

    preview = result.head(
        10
    ).to_string(
        index=False
    )

    prompt = f"""
You are a data analyst.

Question:

{question}

Result:

{preview}

Explain the result simply.

Do not invent information.
Use only the result.
"""

    answer = generate_local_text(
        prompt,
        max_tokens=220,
    )

    if answer:

        return answer

    return (
        f"The query returned "
        f"{len(result):,} rows."
    )


# ============================================================
# AUTOMATIC CHART
# ============================================================

def create_result_chart(
    result,
    question
):

    if result is None:
        return None

    if result.empty:
        return None

    if len(result) > 100:
        return None

    numeric_columns = (
        result.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    if not numeric_columns:
        return None

    if result.shape[1] >= 2:

        x_column = result.columns[0]

        y_column = numeric_columns[0]

        if x_column == y_column:

            if len(numeric_columns) >= 2:

                y_column = numeric_columns[1]

            else:

                return None

        if (
            result[x_column].dtype
            == "object"
            or
            str(
                result[
                    x_column
                ].dtype
            )
            == "category"
        ):

            return px.bar(
                result,
                x=x_column,
                y=y_column,
                title=question,
            )

        if pd.api.types.is_numeric_dtype(
            result[x_column]
        ):

            return px.scatter(
                result,
                x=x_column,
                y=y_column,
                title=question,
            )

    return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:20px;
        ">
            <div style="
                font-size:52px;
                margin-bottom:5px;
            ">
                🤖
            </div>

            <div style="
                font-size:24px;
                font-weight:900;
            ">
                AI Data Analyst
            </div>

            <div style="
                color:#94a3b8;
                margin-top:8px;
            ">
                Automated Data Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
            "json",
            "parquet",
        ],
    )

    if uploaded_file:

        new_filename = (
            uploaded_file.name
        )

        if (
            st.session_state.filename
            != new_filename
        ):

            loaded_df = (
                load_uploaded_file(
                    uploaded_file
                )
            )

            if loaded_df is not None:

                st.session_state.df = (
                    loaded_df
                )

                st.session_state.filename = (
                    new_filename
                )

                st.session_state.ai_report = (
                    None
                )

                st.session_state.ai_context = (
                    None
                )

                st.session_state.chat_history = []

    if st.session_state.df is not None:

        st.markdown("---")

        st.success(
            f"Loaded: "
            f"{st.session_state.filename}"
        )

        st.write(
            f"Rows: "
            f"{len(st.session_state.df):,}"
        )

        st.write(
            f"Columns: "
            f"{len(st.session_state.df.columns):,}"
        )


# ============================================================
# NO DATA
# ============================================================

if st.session_state.df is None:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:100px 20px;
        ">

        <div style="
            font-size:80px;
        ">
        🤖
        </div>

        <h1>
        AI Data Analyst
        </h1>

        <p style="
            color:#94a3b8;
            font-size:20px;
        ">
        Upload a CSV, Excel, JSON or Parquet
        dataset to automatically analyze it.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# DATA
# ============================================================

df = st.session_state.df


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown(
    "### 🧭 Navigation"
)

selected_page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "🔍 EDA",
        "📊 Statistics",
        "📈 Visualizations",
        "💡 AI Insights",
        "💬 Ask Your Data",
    ],
)


# ============================================================
# DASHBOARD
# ============================================================

if selected_page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">'
        '🏠 Data Dashboard'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Automatic overview of your uploaded dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    eda = calculate_eda(
        df
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Rows",
            f"{eda['rows']:,}",
        )

    with c2:

        st.metric(
            "Columns",
            f"{eda['columns']:,}",
        )

    with c3:

        st.metric(
            "Missing Cells",
            f"{eda['missing'].sum():,}",
        )

    with c4:

        st.metric(
            "Duplicate Rows",
            f"{eda['duplicates']:,}",
        )

    st.markdown("---")

    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
    )

    st.subheader(
        "🧬 Column Information"
    )

    st.dataframe(
        pd.DataFrame(
            get_dataset_schema(df)
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# EDA
# ============================================================

elif selected_page == "🔍 EDA":

    st.markdown(
        '<div class="section-title">'
        '🔍 Exploratory Data Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    eda = calculate_eda(
        df
    )

    st.subheader(
        "📌 Data Types"
    )

    type_df = pd.DataFrame(
        {
            "Column":
                df.columns,

            "Data Type":
                [
                    str(
                        df[col].dtype
                    )
                    for col in df.columns
                ],

            "Unique Values":
                [
                    int(
                        df[col].nunique()
                    )
                    for col in df.columns
                ],

            "Missing":
                [
                    int(
                        df[col].isna().sum()
                    )
                    for col in df.columns
                ],
        }
    )

    st.dataframe(
        type_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "⚠️ Missing Values"
    )

    missing_df = pd.DataFrame(
        {
            "Column":
                eda[
                    "missing"
                ].index,

            "Missing":
                eda[
                    "missing"
                ].values,
        }
    )

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )

    if eda["duplicates"]:

        st.warning(
            f"{eda['duplicates']:,} "
            "duplicate rows detected."
        )

    else:

        st.success(
            "No duplicate rows detected."
        )


# ============================================================
# STATISTICS
# ============================================================

elif selected_page == "📊 Statistics":

    st.markdown(
        '<div class="section-title">'
        '📊 Statistical Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    if numeric_columns:

        st.subheader(
            "📐 Descriptive Statistics"
        )

        st.dataframe(
            df[numeric_columns]
            .describe()
            .T,
            use_container_width=True,
        )

        st.subheader(
            "🔗 Correlation Matrix"
        )

        if len(numeric_columns) >= 2:

            corr = df[
                numeric_columns
            ].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                title="Correlation Matrix",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "At least two numeric columns "
                "are required."
            )

    else:

        st.info(
            "No numeric columns detected."
        )


# ============================================================
# VISUALIZATIONS
# ============================================================

elif selected_page == "📈 Visualizations":

    st.markdown(
        '<div class="section-title">'
        '📈 Smart Visualizations'
        '</div>',
        unsafe_allow_html=True,
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns.tolist()
    )

    if numeric_columns:

        selected_numeric = st.selectbox(
            "Select numeric column",
            numeric_columns,
        )

        chart_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Box Plot",
            ],
        )

        if chart_type == "Histogram":

            fig = px.histogram(
                df,
                x=selected_numeric,
                title=(
                    f"Distribution of "
                    f"{selected_numeric}"
                ),
            )

        else:

            fig = px.box(
                df,
                y=selected_numeric,
                title=(
                    f"Box Plot of "
                    f"{selected_numeric}"
                ),
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    if categorical_columns:

        st.markdown("---")

        selected_category = st.selectbox(
            "Select categorical column",
            categorical_columns,
        )

        counts = (
            df[
                selected_category
            ]
            .value_counts()
            .head(20)
            .reset_index()
        )

        counts.columns = [
            selected_category,
            "count",
        ]

        fig = px.bar(
            counts,
            x=selected_category,
            y="count",
            title=(
                f"Distribution of "
                f"{selected_category}"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# AI INSIGHTS
# ============================================================

elif selected_page == "💡 AI Insights":

    st.markdown(
        '<div class="section-title">'
        '💡 AI Data Insights'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Python performs the statistical "
        "analysis first. The local AI model "
        "then explains the verified findings."
    )

    if st.button(
        "🧠 Analyze Dataset",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Analyzing dataset..."
        ):

            report, context = (
                generate_ai_insights(
                    df
                )
            )

            st.session_state.ai_report = (
                report
            )

            st.session_state.ai_context = (
                context
            )

    if st.session_state.ai_report:

        st.markdown("---")

        st.markdown(
            st.session_state.ai_report
        )

        context = (
            st.session_state.ai_context
        )

        if context:

            st.markdown("---")

            st.subheader(
                "🔎 Verified Analysis"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Rows",
                    f"{context['rows']:,}",
                )

            with c2:

                st.metric(
                    "Columns",
                    f"{context['columns']:,}",
                )

            with c3:

                st.metric(
                    "Missing %",
                    f"{context['missing_percentage']}%",
                )

            with c4:

                st.metric(
                    "Duplicates",
                    f"{context['duplicate_rows']:,}",
                )

            if context[
                "correlations"
            ]:

                st.subheader(
                    "🔗 Strong Relationships"
                )

                st.dataframe(
                    pd.DataFrame(
                        context[
                            "correlations"
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            if context[
                "outliers"
            ]:

                st.subheader(
                    "🚨 Potential Outliers"
                )

                st.dataframe(
                    pd.DataFrame(
                        context[
                            "outliers"
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    else:

        st.info(
            "Click Analyze Dataset to generate insights."
        )


# ============================================================
# ASK YOUR DATA
# ============================================================

elif selected_page == "💬 Ask Your Data":

    st.markdown(
        '<div class="section-title">'
        '💬 Ask Your Data'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Ask questions about your dataset using natural language.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "🧬 Dataset Schema"
    ):

        st.dataframe(
            pd.DataFrame(
                get_dataset_schema(
                    df
                )
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "💡 Example Questions"
    )

    examples = [
        "How many rows are in the dataset?",
        "What is the average salary?",
        "What is the highest price?",
        "What is the minimum age?",
        "Show unique values of category",
        "Show the data",
    ]

    example_columns = st.columns(3)

    for i, example in enumerate(
        examples
    ):

        with example_columns[
            i % 3
        ]:

            if st.button(
                example,
                key=f"example_{i}",
                use_container_width=True,
            ):

                st.session_state[
                    "pending_question"
                ] = example

    pending_question = (
        st.session_state.pop(
            "pending_question",
            None,
        )
    )

    question = st.chat_input(
        "Ask something about your dataset..."
    )

    if pending_question:

        question = pending_question

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    for message in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message.get(
                "sql"
            ):

                with st.expander(
                    "🔍 Generated SQL"
                ):

                    st.code(
                        message["sql"],
                        language="sql",
                    )

            if message.get(
                "data"
            ) is not None:

                st.dataframe(
                    message["data"],
                    use_container_width=True,
                )

    # --------------------------------------------------------
    # NEW QUESTION
    # --------------------------------------------------------

    if question:

        st.session_state.chat_history.append(
            {
                "role":
                    "user",

                "content":
                    question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Analyzing your question..."
            ):

                sql = (
                    generate_sql_from_question(
                        question,
                        df,
                    )
                )

                if not sql:

                    answer = (
                        "I could not understand "
                        "the question well enough "
                        "to create a safe SQL query."
                    )

                    st.warning(
                        answer
                    )

                    st.session_state.chat_history.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                answer,

                            "sql":
                                None,

                            "data":
                                None,
                        }
                    )

                else:

                    sql = (
                        clean_generated_sql(
                            sql
                        )
                    )

                    valid, error = (
                        validate_generated_sql(
                            sql,
                            df,
                        )
                    )

                    if not valid:

                        st.warning(
                            f"SQL rejected: {error}"
                        )

                    else:

                        result, error = (
                            execute_sql(
                                df,
                                sql,
                            )
                        )

                        if error:

                            st.error(
                                "SQL execution failed."
                            )

                            st.code(
                                sql,
                                language="sql",
                            )

                            st.error(
                                error
                            )

                        else:

                            answer = (
                                explain_query_result(
                                    question,
                                    result,
                                )
                            )

                            st.markdown(
                                answer
                            )

                            if result is not None:

                                st.dataframe(
                                    result,
                                    use_container_width=True,
                                )

                                fig = (
                                    create_result_chart(
                                        result,
                                        question,
                                    )
                                )

                                if fig:

                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                    )

                            with st.expander(
                                "🔍 Generated SQL"
                            ):

                                st.code(
                                    sql,
                                    language="sql",
                                )

                            st.session_state.chat_history.append(
                                {
                                    "role":
                                        "assistant",

                                    "content":
                                        answer,

                                    "sql":
                                        sql,

                                    "data":
                                        result,
                                }
                            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🤖 AI Data Analyst | "
    "Python • Statistics • EDA • ML • NLP/LLM • "
    "Data Engineering • Visualization • Streamlit"
)