# src/ai/insight_generator.py

from .llm_engine import LocalLLM
from .prompt_builder import build_insight_prompt


class InsightGenerator:

    def __init__(self):
        self.llm = None

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        if self.llm is None:
            self.llm = LocalLLM()

    # ========================================================
    # GENERATE INSIGHTS
    # ========================================================

    def generate(
        self,
        df,
        profile,
        patterns,
        ml_analysis,
    ):
        """
        Generate AI-powered insights from
        the analytical results.
        """

        self.load_model()

        prompt = build_insight_prompt(
            df=df,
            profile=profile,
            patterns=patterns,
            ml_analysis=ml_analysis,
        )

        response = self.llm.generate(
            prompt
        )

        if response is None:
            return (
                "Unable to generate AI insights."
            )

        return str(response)