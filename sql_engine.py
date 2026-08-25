# ============================================================
# sql_engine.py
# RELIABLE ASK YOUR DATA ENGINE
# ============================================================

import re
import numpy as np
import pandas as pd


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_text(text):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower()
    )


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    question,
    columns
):

    question_normalized = normalize_text(
        question
    )

    candidates = []

    for column in columns:

        normalized = normalize_text(
            column
        )

        if not normalized:
            continue

        if normalized in question_normalized:

            candidates.append(
                (
                    len(normalized),
                    column
                )
            )

    if candidates:

        candidates.sort(
            reverse=True
        )

        return candidates[0][1]

    return None


# ============================================================
# SQL QUOTE
# ============================================================

def quote_identifier(
    column
):

    return (
        '"'
        +
        str(column).replace(
            '"',
            '""'
        )
        +
        '"'
    )


# ============================================================
# RULE-BASED QUESTION ENGINE
# ============================================================

def answer_question_directly(
    question,
    df
):

    q = question.lower().strip()

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    columns = list(
        df.columns
    )

    # ========================================================
    # ROW COUNT
    # ========================================================

    if (
        "how many rows" in q
        or
        "number of rows" in q
        or
        "row count" in q
        or
        "how many records" in q
        or
        "number of records" in q
        or
        "total records" in q
    ):

        result = pd.DataFrame(
            {
                "Row Count":
                    [len(df)]
            }
        )

        return (
            True,
            f"The dataset contains **{len(df):,} rows**.",
            result,
            None
        )

    # ========================================================
    # COLUMN COUNT
    # ========================================================

    if (
        "how many columns" in q
        or
        "number of columns" in q
        or
        "column count" in q
    ):

        result = pd.DataFrame(
            {
                "Column Count":
                    [len(df.columns)]
            }
        )

        return (
            True,
            f"The dataset contains **{len(df.columns):,} columns**.",
            result,
            None
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    if (
        "missing values" in q
        or
        "missing data" in q
        or
        "null values" in q
        or
        "nulls" in q
    ):

        missing = (
            df.isna()
            .sum()
            .reset_index()
        )

        missing.columns = [
            "Column",
            "Missing Values"
        ]

        missing = missing[
            missing[
                "Missing Values"
            ] > 0
        ]

        if missing.empty:

            answer = (
                "✅ There are **no missing values** "
                "in the dataset."
            )

        else:

            total = int(
                missing[
                    "Missing Values"
                ].sum()
            )

            answer = (
                f"⚠️ There are **{total:,} "
                f"missing cells** across "
                f"{len(missing)} columns."
            )

        return (
            True,
            answer,
            missing,
            None
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    if (
        "duplicate" in q
        or
        "duplicates" in q
    ):

        count = int(
            df.duplicated().sum()
        )

        result = pd.DataFrame(
            {
                "Duplicate Rows":
                    [count]
            }
        )

        if count:

            answer = (
                f"⚠️ There are **{count:,} "
                f"duplicate rows**."
            )

        else:

            answer = (
                "✅ No duplicate rows were detected."
            )

        return (
            True,
            answer,
            result,
            None
        )

    # ========================================================
    # COLUMN LIST
    # ========================================================

    if (
        "what columns" in q
        or
        "list columns" in q
        or
        "column names" in q
        or
        "show columns" in q
    ):

        result = pd.DataFrame(
            {
                "Columns":
                    columns
            }
        )

        return (
            True,
            f"The dataset has **{len(columns)} columns**.",
            result,
            None
        )

    # ========================================================
    # DATA TYPES
    # ========================================================

    if (
        "data types" in q
        or
        "column types" in q
        or
        "types of columns" in q
    ):

        result = pd.DataFrame(
            {
                "Column":
                    columns,

                "Data Type":
                    [
                        str(
                            df[c].dtype
                        )
                        for c in columns
                    ]
            }
        )

        return (
            True,
            "Here are the detected column data types.",
            result,
            None
        )

    # ========================================================
    # AVERAGE / MEAN
    # ========================================================

    if (
        "average" in q
        or
        "mean" in q
    ):

        column = find_column(
            q,
            numeric
        )

        if column:

            value = df[column].mean()

            result = pd.DataFrame(
                {
                    "Column":
                        [column],

                    "Average":
                        [value]
                }
            )

            return (
                True,
                f"The average **{column}** is "
                f"**{value:,.2f}**.",
                result,
                None
            )

    # ========================================================
    # MEDIAN
    # ========================================================

    if "median" in q:

        column = find_column(
            q,
            numeric
        )

        if column:

            value = df[column].median()

            result = pd.DataFrame(
                {
                    "Column":
                        [column],

                    "Median":
                        [value]
                }
            )

            return (
                True,
                f"The median **{column}** is "
                f"**{value:,.2f}**.",
                result,
                None
            )

    # ========================================================
    # SUM / TOTAL
    # ========================================================

    if (
        "sum" in q
        or
        "total" in q
    ):

        column = find_column(
            q,
            numeric
        )

        if column:

            value = df[column].sum()

            result = pd.DataFrame(
                {
                    "Column":
                        [column],

                    "Total":
                        [value]
                }
            )

            return (
                True,
                f"The total **{column}** is "
                f"**{value:,.2f}**.",
                result,
                None
            )

    # ========================================================
    # MAXIMUM
    # ========================================================

    if any(
        word in q
        for word in [
            "highest",
            "maximum",
            "max value",
            "largest"
        ]
    ):

        column = find_column(
            q,
            numeric
        )

        if column:

            value = df[column].max()

            result = pd.DataFrame(
                {
                    "Column":
                        [column],

                    "Maximum":
                        [value]
                }
            )

            return (
                True,
                f"The maximum **{column}** is "
                f"**{value:,.2f}**.",
                result,
                None
            )

    # ========================================================
    # MINIMUM
    # ========================================================

    if any(
        word in q
        for word in [
            "lowest",
            "minimum",
            "min value",
            "smallest"
        ]
    ):

        column = find_column(
            q,
            numeric
        )

        if column:

            value = df[column].min()

            result = pd.DataFrame(
                {
                    "Column":
                        [column],

                    "Minimum":
                        [value]
                }
            )

            return (
                True,
                f"The minimum **{column}** is "
                f"**{value:,.2f}**.",
                result,
                None
            )

    # ========================================================
    # UNIQUE VALUES
    # ========================================================

    if (
        "unique values" in q
        or
        "unique categories" in q
        or
        "distinct values" in q
    ):

        column = find_column(
            q,
            columns
        )

        if column:

            result = (
                df[column]
                .value_counts(
                    dropna=False
                )
                .reset_index()
            )

            result.columns = [
                column,
                "Count"
            ]

            result = result.head(
                50
            )

            return (
                True,
                f"Here are the most common values "
                f"for **{column}**.",
                result,
                None
            )

    # ========================================================
    # TOP / MOST COMMON
    # ========================================================

    if (
        "most common" in q
        or
        "top categories" in q
        or
        "most frequent" in q
    ):

        column = find_column(
            q,
            columns
        )

        if column:

            result = (
                df[column]
                .value_counts()
                .head(20)
                .reset_index()
            )

            result.columns = [
                column,
                "Count"
            ]

            return (
                True,
                f"Here are the most frequent "
                f"values in **{column}**.",
                result,
                None
            )

    # ========================================================
    # SHOW DATA
    # ========================================================

    if (
        "show data" in q
        or
        "show rows" in q
        or
        "show records" in q
        or
        "preview" in q
    ):

        result = df.head(20)

        return (
            True,
            "Here is a preview of the dataset.",
            result,
            None
        )

    return (
        False,
        None,
        None,
        None
    )


# ============================================================
# GENERATE SQL USING LLM
# ============================================================

def generate_ai_sql(
    question,
    df,
    llm
):

    if llm is None:

        return None

    from ai_engine import generate_text

    schema = "\n".join(
        f"- {col}: {df[col].dtype}"
        for col in df.columns
    )

    prompt = f"""
You are a DuckDB SQL generator.

Table name:
dataset

Schema:
{schema}

Question:
{question}

Rules:

Return ONLY SQL.

Use SELECT only.

Never modify data.

Use the exact column names.

Use DuckDB syntax.

SQL:
"""

    sql = generate_text(
        llm,
        prompt,
        max_tokens=200
    )

    if not sql:
        return None

    sql = sql.replace(
        "```sql",
        ""
    )

    sql = sql.replace(
        "```",
        ""
    )

    match = re.search(
        r"(SELECT|WITH)\b",
        sql,
        flags=re.I
    )

    if match:

        sql = sql[
            match.start():
        ]

    return sql.strip().rstrip(";").strip()


# ============================================================
# VALIDATE SQL
# ============================================================

def validate_sql(
    sql
):

    if not sql:

        return (
            False,
            "No SQL generated."
        )

    q = sql.lower().strip()

    if not (
        q.startswith("select")
        or
        q.startswith("with")
    ):

        return (
            False,
            "Only SELECT queries are allowed."
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
        "copy"
    ]

    for word in forbidden:

        if re.search(
            rf"\b{word}\b",
            q
        ):

            return (
                False,
                f"Unsafe SQL operation detected: {word}"
            )

    return (
        True,
        "Valid"
    )


# ============================================================
# EXECUTE SQL
# ============================================================

def execute_query(
    df,
    sql
):

    try:

        import duckdb

        connection = duckdb.connect(
            ":memory:"
        )

        connection.register(
            "dataset",
            df
        )

        result = connection.execute(
            sql
        ).df()

        connection.close()

        return (
            result,
            None
        )

    except Exception as e:

        return (
            None,
            str(e)
        )