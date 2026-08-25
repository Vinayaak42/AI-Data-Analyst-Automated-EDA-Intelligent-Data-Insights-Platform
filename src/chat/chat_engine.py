import re

from .query_engine import QueryEngine
from .query_generator import QueryGenerator


class ChatEngine:

    def __init__(self, df):

        self.df = df

        self.query_engine = QueryEngine(
            df
        )

        self.query_generator = (
            QueryGenerator()
        )

    def validate_sql(self, sql):

        sql_clean = sql.strip().lower()

        # Only SELECT / WITH queries
        if not (
            sql_clean.startswith("select")
            or sql_clean.startswith("with")
        ):

            return False

        forbidden = [
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "truncate",
            "attach",
            "copy",
            "install",
            "load",
            "pragma",
        ]

        for keyword in forbidden:

            pattern = rf"\b{keyword}\b"

            if re.search(
                pattern,
                sql_clean,
            ):

                return False

        return True

    def ask(self, question):

        sql = (
            self.query_generator.generate_sql(
                question,
                self.df,
            )
        )

        if sql == "CANNOT_ANSWER":

            return {
                "success": False,
                "message": (
                    "I cannot answer this "
                    "question using the uploaded dataset."
                ),
            }

        if not self.validate_sql(sql):

            return {
                "success": False,
                "message": (
                    "The generated query "
                    "failed the safety check."
                ),
            }

        try:

            result = (
                self.query_engine.execute(
                    sql
                )
            )

            return {
                "success": True,
                "sql": sql,
                "result": result,
            }

        except Exception as e:

            return {
                "success": False,
                "sql": sql,
                "message": str(e),
            }