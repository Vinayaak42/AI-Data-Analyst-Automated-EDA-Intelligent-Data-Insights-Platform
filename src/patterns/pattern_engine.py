from .trend_detector import (
    detect_numeric_trends,
)

from .correlation_detector import (
    detect_correlations,
)

from .anomaly_detector import (
    detect_anomalies,
)

from .category_detector import (
    detect_category_patterns,
)


def generate_patterns(df):

    return {
        "trends": detect_numeric_trends(df),

        "correlations": detect_correlations(df),

        "anomalies": detect_anomalies(df),

        "categories": detect_category_patterns(df),
    }