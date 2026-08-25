from src.ai.llm_engine import LocalLLM


class QueryGenerator:

    def __init__(self):

        self.llm = LocalLLM()

    def build_schema_context(
        self,
        df,
    ):

        lines = []

        for column, dtype in zip(
            df.columns,
            df.dtypes,
        ):

            lines.append(
                f"- {column}: {dtype}"
            )

        return "\n".join(lines)

    def generate_sql(
        self,
        question,
        df,
    ):

        schema = self.build_schema_context(
            df
        )

        prompt = f"""
You are a data analyst.

Convert the user's question into DuckDB SQL.

The dataframe is registered as:

dataset

DATABASE SCHEMA
---------------
{schema}

USER QUESTION
-------------
{question}

RULES
-----

1. Use only columns from the schema.
2. Query only the table `dataset`.
3. Generate SELECT queries only.
4. Do not modify data.
5. Do not use INSERT.
6. Do not use UPDATE.
7. Do not use DELETE.
8. Do not use DROP.
9. Do not use ALTER.
10. Return only SQL.
11. Do not use markdown.
12. If the question cannot be answered from the dataset,
    return:

CANNOT_ANSWER

SQL:
"""

        sql = self.llm.generate(
            prompt,
            max_new_tokens=250,
        )

        sql = sql.strip()

        if sql.startswith("```"):

            sql = sql.replace(
                "```sql",
                ""
            )

            sql = sql.replace(
                "```",
                ""
            )

            sql = sql.strip()

        return sql