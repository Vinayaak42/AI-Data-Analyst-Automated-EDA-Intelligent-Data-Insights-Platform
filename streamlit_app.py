
# ============================================================
# IMPORTS
# ============================================================

import os
import io
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# INTERNAL MODULES
# ============================================================

try:

    from ai_engine import (
        load_local_llm,
        generate_ai_insights,
    )

    AI_ENGINE_AVAILABLE = True

except Exception as e:

    AI_ENGINE_AVAILABLE = False
    AI_ENGINE_ERROR = str(e)


try:

    from sql_engine import (
        answer_question_directly,
        generate_ai_sql,
        validate_sql,
        execute_query,
    )

    SQL_ENGINE_AVAILABLE = True

except Exception as e:

    SQL_ENGINE_AVAILABLE = False
    SQL_ENGINE_ERROR = str(e)


# ============================================================
# PAGE CONFIGURATION
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

    /* ========================================================
       MAIN
       ======================================================== */

    .main {
        padding-top: 1rem;
    }

    /* ========================================================
       TITLE
       ======================================================== */

    .main-title {

        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;

    }

    .subtitle {

        font-size: 17px;
        color: #777;
        margin-bottom: 25px;

    }

    /* ========================================================
       SECTION
       ======================================================== */

    .section-title {

        font-size: 30px;
        font-weight: 700;
        margin-bottom: 5px;

    }

    .section-subtitle {

        color: #777;
        font-size: 15px;
        margin-bottom: 20px;

    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {

        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.04);

    }

    /* ========================================================
       INSIGHT CARD
       ======================================================== */

    .insight-card {

        padding: 15px;
        margin-bottom: 10px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);

    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {

        border-right: 1px solid rgba(128,128,128,0.15);

    }

    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {

        border-radius: 8px;

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


if "llm" not in st.session_state:

    st.session_state.llm = None


if "ask_history" not in st.session_state:

    st.session_state.ask_history = []


if "filters" not in st.session_state:

    st.session_state.filters = {}


if "analysis_cache" not in st.session_state:

    st.session_state.analysis_cache = {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def clean_dataframe(df):

    """
    Basic dataframe cleanup.
    """

    df = df.copy()

    # Remove completely empty rows

    df = df.dropna(
        how="all"
    )

    # Remove completely empty columns

    df = df.dropna(
        axis=1,
        how="all"
    )

    # Clean column names

    new_columns = []

    for i, col in enumerate(df.columns):

        name = str(col).strip()

        if not name:

            name = f"Column_{i+1}"

        new_columns.append(name)

    df.columns = new_columns

    return df


# ============================================================
# FILE LOADER
# ============================================================


def load_uploaded_file(uploaded_file):

    """
    Load CSV / Excel / JSON / Parquet.
    """

    filename = uploaded_file.name.lower()

    try:

        if filename.endswith(".csv"):

            try:

                df = pd.read_csv(
                    uploaded_file
                )

            except UnicodeDecodeError:

                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    encoding="latin1"
                )

        elif (
            filename.endswith(".xlsx")
            or filename.endswith(".xls")
        ):

            df = pd.read_excel(
                uploaded_file
            )

        elif filename.endswith(".json"):

            df = pd.read_json(
                uploaded_file
            )

        elif filename.endswith(".parquet"):

            df = pd.read_parquet(
                uploaded_file
            )

        else:

            return (
                None,
                "Unsupported file format."
            )

        df = clean_dataframe(
            df
        )

        return (
            df,
            None
        )

    except Exception as e:

        return (
            None,
            str(e)
        )


# ============================================================
# FORMAT NUMBER
# ============================================================


def format_number(value):

    try:

        if pd.isna(value):

            return "N/A"

        if isinstance(
            value,
            (int, np.integer)
        ):

            return f"{value:,}"

        if isinstance(
            value,
            (float, np.floating)
        ):

            return f"{value:,.2f}"

        return str(value)

    except Exception:

        return str(value)


# ============================================================
# DATA PROFILE
# ============================================================


def create_profile(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=[
            "datetime",
            "datetimetz"
        ]
    ).columns.tolist()

    missing = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    memory = (
        df.memory_usage(
            deep=True
        ).sum()
        /
        1024**2
    )

    return {

        "rows":
            len(df),

        "columns":
            len(df.columns),

        "numeric":
            len(numeric_columns),

        "categorical":
            len(categorical_columns),

        "datetime":
            len(datetime_columns),

        "missing":
            missing,

        "duplicates":
            duplicates,

        "memory":
            memory,

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "datetime_columns":
            datetime_columns,
    }


# ============================================================
# HEADER
# ============================================================


st.markdown(
    '<div class="main-title">🤖 AI Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload your dataset and automatically discover
    patterns, statistics, relationships, anomalies,
    visualizations and AI-powered insights.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    st.title("🤖 AI Data Analyst")

    st.caption(
        "Automated EDA + AI + ML"
    )

    st.divider()

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(

        "📂 Upload Dataset",

        type=[
            "csv",
            "xlsx",
            "xls",
            "json",
            "parquet"
        ],

        help=(
            "Supported formats: "
            "CSV, Excel, JSON and Parquet"
        )
    )

    # ========================================================
    # LOAD FILE
    # ========================================================

    if uploaded_file is not None:

        # Only reload when file changes

        if (
            st.session_state.filename
            != uploaded_file.name
        ):

            with st.spinner(
                "Loading dataset..."
            ):

                new_df, error = (
                    load_uploaded_file(
                        uploaded_file
                    )
                )

            if error:

                st.error(
                    f"❌ {error}"
                )

            else:

                st.session_state.df = (
                    new_df
                )

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.session_state.ai_report = None

                st.session_state.ask_history = []

                st.session_state.analysis_cache = {}

                st.success(
                    "Dataset loaded successfully."
                )

    st.divider()

    # ========================================================
    # NAVIGATION
    # ========================================================

    pages = [

        "🏠 Dashboard",

        "📋 Data Preview",

        "📊 EDA",

        "📈 Visualizations",

        "💡 AI Insights",

        "💬 Ask Your Data",

        "🤖 ML Readiness",

        "⚙️ Data Quality",

    ]

    page = st.radio(
        "Navigation",
        pages
    )

    st.divider()

    # ========================================================
    # DATASET STATUS
    # ========================================================

    if st.session_state.df is not None:

        st.success(
            f"📁 {st.session_state.filename}"
        )

        st.caption(
            f"{len(st.session_state.df):,} rows × "
            f"{len(st.session_state.df.columns):,} columns"
        )

    else:

        st.warning(
            "Upload a dataset to begin."
        )


# ============================================================
# CHECK DATASET
# ============================================================


df = st.session_state.df


if df is None:

    st.info(
        """
        ### 👋 Welcome to AI Data Analyst

        Start by uploading a dataset from the sidebar.

        Supported:

        - CSV
        - Excel
        - JSON
        - Parquet

        The application will automatically perform:

        **EDA → Statistics → Patterns → Anomalies → Charts → AI Insights → ML Readiness**
        """
    )

    st.stop()


# ============================================================
# PROFILE
# ============================================================


profile = create_profile(
    df
)


# ============================================================
# DASHBOARD
# ============================================================


if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">🏠 Dataset Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'High-level overview of your dataset'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Rows",
        f"{profile['rows']:,}"
    )

    c2.metric(
        "Columns",
        f"{profile['columns']:,}"
    )

    c3.metric(
        "Numeric",
        profile["numeric"]
    )

    c4.metric(
        "Missing Cells",
        f"{profile['missing']:,}"
    )

    c5.metric(
        "Duplicates",
        f"{profile['duplicates']:,}"
    )

    st.divider()

    # ========================================================
    # DATA TYPES
    # ========================================================

    left, right = st.columns(2)

    with left:

        st.subheader(
            "📊 Column Types"
        )

        type_counts = pd.DataFrame(
            {
                "Type": [
                    "Numeric",
                    "Categorical",
                    "Datetime"
                ],

                "Count": [
                    profile["numeric"],
                    profile["categorical"],
                    profile["datetime"]
                ]
            }
        )

        fig = px.bar(
            type_counts,
            x="Type",
            y="Count",
            title="Column Type Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader(
            "🧹 Data Quality"
        )

        quality_data = pd.DataFrame(
            {
                "Metric": [
                    "Missing",
                    "Duplicates",
                    "Complete Rows"
                ],

                "Count": [

                    profile["missing"],

                    profile["duplicates"],

                    int(
                        len(df)
                        -
                        profile["duplicates"]
                    )
                ]
            }
        )

        fig = px.bar(
            quality_data,
            x="Metric",
            y="Count",
            title="Data Quality Overview"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # NUMERIC SUMMARY
    # ========================================================

    st.subheader(
        "🔢 Numeric Summary"
    )

    numeric = df.select_dtypes(
        include=np.number
    )

    if not numeric.empty:

        st.dataframe(
            numeric.describe().T,
            use_container_width=True
        )

    else:

        st.info(
            "No numeric columns detected."
        )


# ============================================================
# DATA PREVIEW
# ============================================================


elif page == "📋 Data Preview":

    st.markdown(
        '<div class="section-title">📋 Data Preview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore and filter your uploaded dataset'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # SEARCH
    # ========================================================

    search = st.text_input(
        "🔎 Search dataset",
        placeholder="Type text to search..."
    )

    filtered_df = df.copy()

    if search:

        mask = filtered_df.astype(
            str
        ).apply(
            lambda column:
            column.str.contains(
                search,
                case=False,
                na=False
            )
        ).any(
            axis=1
        )

        filtered_df = filtered_df[
            mask
        ]

    # ========================================================
    # ROW LIMIT
    # ========================================================

    row_limit = st.slider(
        "Rows to display",
        min_value=10,
        max_value=min(
            1000,
            max(10, len(filtered_df))
        ),
        value=min(
            100,
            max(10, len(filtered_df))
        )
    )

    st.caption(
        f"Showing {min(row_limit, len(filtered_df)):,} "
        f"of {len(filtered_df):,} matching rows."
    )

    st.dataframe(
        filtered_df.head(row_limit),
        use_container_width=True,
        height=500
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv_data = filtered_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download Current Data",
        csv_data,
        "filtered_dataset.csv",
        "text/csv"
    )


# ============================================================
# EDA
# ============================================================


elif page == "📊 EDA":

    st.markdown(
        '<div class="section-title">📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Automatically analyze distributions, missing values, '
        'outliers and relationships'
        '</div>',
        unsafe_allow_html=True
    )

    numeric = df.select_dtypes(
        include=np.number
    )

    categorical = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )

    # ========================================================
    # NUMERIC STATISTICS
    # ========================================================

    st.subheader(
        "🔢 Statistical Summary"
    )

    if not numeric.empty:

        stats = numeric.describe().T

        stats[
            "missing"
        ] = numeric.isna().sum()

        stats[
            "unique"
        ] = numeric.nunique()

        st.dataframe(
            stats,
            use_container_width=True
        )

    else:

        st.info(
            "No numeric columns available."
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    st.subheader(
        "⚠️ Missing Values"
    )

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_df = pd.DataFrame(
        {
            "Column":
                missing.index,

            "Missing":
                missing.values,

            "Percentage":
                (
                    missing.values
                    /
                    len(df)
                    *
                    100
                )
        }
    )

    missing_df = missing_df[
        missing_df["Missing"] > 0
    ]

    if missing_df.empty:

        st.success(
            "✅ No missing values detected."
        )

    else:

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        fig = px.bar(
            missing_df,
            x="Column",
            y="Percentage",
            title="Missing Values by Column"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    st.subheader(
        "♻️ Duplicate Rows"
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count:

        st.warning(
            f"{duplicate_count:,} duplicate rows detected."
        )

    else:

        st.success(
            "No duplicate rows detected."
        )

    # ========================================================
    # OUTLIERS
    # ========================================================

    st.subheader(
        "🚨 Outlier Detection"
    )

    if not numeric.empty:

        outlier_rows = []

        for column in numeric.columns:

            series = (
                numeric[column]
                .dropna()
            )

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

            percentage = (
                count
                /
                len(series)
                *
                100
            )

            outlier_rows.append(
                {
                    "Column":
                        column,

                    "Outliers":
                        count,

                    "Percentage":
                        percentage,

                    "Lower Bound":
                        lower,

                    "Upper Bound":
                        upper,
                }
            )

        if outlier_rows:

            outlier_df = pd.DataFrame(
                outlier_rows
            )

            st.dataframe(
                outlier_df,
                use_container_width=True
            )

        else:

            st.info(
                "No suitable numeric columns for outlier analysis."
            )

    # ========================================================
    # CORRELATION
    # ========================================================

    st.subheader(
        "🔗 Correlation Analysis"
    )

    if numeric.shape[1] >= 2:

        corr = numeric.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "At least two numeric columns are required."
        )

    # ========================================================
    # CATEGORICAL ANALYSIS
    # ========================================================

    st.subheader(
        "🏷️ Categorical Analysis"
    )

    if not categorical.empty:

        selected_cat = st.selectbox(
            "Select categorical column",
            categorical.columns
        )

        counts = (
            df[selected_cat]
            .value_counts(
                dropna=False
            )
            .head(20)
            .reset_index()
        )

        counts.columns = [
            selected_cat,
            "Count"
        ]

        fig = px.bar(
            counts,
            x=selected_cat,
            y="Count",
            title=f"Top Values - {selected_cat}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            counts,
            use_container_width=True
        )

    else:

        st.info(
            "No categorical columns detected."
        )


# ============================================================
# VISUALIZATIONS
# ============================================================


elif page == "📈 Visualizations":

    st.markdown(
        '<div class="section-title">📈 Interactive Visualizations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Create charts automatically from your dataset'
        '</div>',
        unsafe_allow_html=True
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    # ========================================================
    # CHART TYPE
    # ========================================================

    chart_type = st.selectbox(
        "📊 Select Chart",
        [
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "Bar Chart",
            "Line Chart",
            "Pie Chart",
            "Correlation Heatmap"
        ]
    )

    # ========================================================
    # HISTOGRAM
    # ========================================================

    if chart_type == "Histogram":

        if numeric_columns:

            column = st.selectbox(
                "Select numeric column",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=column,
                marginal="box",
                title=f"Distribution of {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No numeric columns found."
            )

    # ========================================================
    # BOX PLOT
    # ========================================================

    elif chart_type == "Box Plot":

        if numeric_columns:

            column = st.selectbox(
                "Select numeric column",
                numeric_columns
            )

            fig = px.box(
                df,
                y=column,
                points="outliers",
                title=f"Box Plot - {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No numeric columns found."
            )

    # ========================================================
    # SCATTER
    # ========================================================

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) >= 2:

            x = st.selectbox(
                "X-axis",
                numeric_columns,
                key="scatter_x"
            )

            y = st.selectbox(
                "Y-axis",
                numeric_columns,
                index=1 if len(numeric_columns) > 1 else 0,
                key="scatter_y"
            )

            color = None

            if categorical_columns:

                use_color = st.checkbox(
                    "Color by category"
                )

                if use_color:

                    color = st.selectbox(
                        "Category",
                        categorical_columns
                    )

            fig = px.scatter(
                df,
                x=x,
                y=y,
                color=color,
                title=f"{y} vs {x}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numeric columns are required."
            )

    # ========================================================
    # BAR
    # ========================================================

    elif chart_type == "Bar Chart":

        if categorical_columns:

            category = st.selectbox(
                "Category",
                categorical_columns
            )

            counts = (
                df[category]
                .value_counts()
                .head(20)
                .reset_index()
            )

            counts.columns = [
                category,
                "Count"
            ]

            fig = px.bar(
                counts,
                x=category,
                y="Count",
                title=f"{category} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No categorical columns found."
            )

    # ========================================================
    # LINE
    # ========================================================

    elif chart_type == "Line Chart":

        if numeric_columns:

            x_options = list(
                df.columns
            )

            x = st.selectbox(
                "X-axis",
                x_options
            )

            y = st.selectbox(
                "Y-axis",
                numeric_columns
            )

            fig = px.line(
                df,
                x=x,
                y=y,
                title=f"{y} over {x}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No numeric columns found."
            )

    # ========================================================
    # PIE
    # ========================================================

    elif chart_type == "Pie Chart":

        if categorical_columns:

            column = st.selectbox(
                "Category",
                categorical_columns
            )

            counts = (
                df[column]
                .value_counts()
                .head(10)
                .reset_index()
            )

            counts.columns = [
                column,
                "Count"
            ]

            fig = px.pie(
                counts,
                names=column,
                values="Count",
                title=f"{column} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No categorical columns found."
            )

    # ========================================================
    # HEATMAP
    # ========================================================

    elif chart_type == "Correlation Heatmap":

        if len(numeric_columns) >= 2:

            corr = df[
                numeric_columns
            ].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numeric columns are required."
            )


# ============================================================
# AI INSIGHTS
# ============================================================


elif page == "💡 AI Insights":

    st.markdown(
        '<div class="section-title">💡 AI-Powered Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Automatic statistical analysis + AI interpretation'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        The system first performs verified statistical
        analysis using Python. The local AI model is used
        only to interpret those verified findings.
        """
    )

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🧠 Analyze Dataset",
        type="primary",
        use_container_width=True
    ):

        if not AI_ENGINE_AVAILABLE:

            st.error(
                "ai_engine.py could not be imported."
            )

            st.code(
                AI_ENGINE_ERROR
            )

        else:

            with st.spinner(
                "Running complete AI analysis..."
            ):

                try:

                    # Load model lazily

                    if (
                        st.session_state.llm
                        is None
                    ):

                        st.session_state.llm = (
                            load_local_llm()
                        )

                    result = (
                        generate_ai_insights(
                            df,
                            st.session_state.llm
                        )
                    )

                    st.session_state.ai_report = (
                        result
                    )

                    st.success(
                        "✅ Analysis completed successfully."
                    )

                except Exception as e:

                    st.error(
                        "AI analysis failed."
                    )

                    st.exception(
                        e
                    )

    # ========================================================
    # REPORT
    # ========================================================

    if st.session_state.ai_report:

        report = (
            st.session_state.ai_report
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        st.subheader(
            "🧠 Executive Summary"
        )

        summary = report.get(
            "summary",
            []
        )

        if isinstance(
            summary,
            list
        ):

            for item in summary:

                st.markdown(
                    item
                )

        else:

            st.markdown(
                str(summary)
            )

        st.divider()

        # ====================================================
        # VERIFIED INSIGHTS
        # ====================================================

        st.subheader(
            "📊 Verified Data Findings"
        )

        insights = report.get(
            "insights",
            []
        )

        for insight in insights:

            st.markdown(
                f"""
                <div class="insight-card">
                {insight}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        report_text = (
            "AI DATA ANALYST REPORT\n"
            "======================\n\n"
        )

        for insight in insights:

            report_text += (
                str(insight)
                +
                "\n\n"
            )

        st.download_button(
            "⬇️ Download Insights Report",
            report_text,
            "ai_data_analysis.txt",
            "text/plain",
            use_container_width=True
        )

    else:

        st.warning(
            "Click **Analyze Dataset** to generate insights."
        )


# ============================================================
# ASK YOUR DATA
# ============================================================


elif page == "💬 Ask Your Data":

    st.markdown(
        '<div class="section-title">💬 Ask Your Data</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Ask questions about your dataset using natural language'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        Common questions are answered directly with Python.
        More advanced questions can use the local Hugging Face
        model to generate DuckDB SQL.
        """
    )

    # ========================================================
    # EXAMPLES
    # ========================================================

    st.subheader(
        "💡 Example Questions"
    )

    examples = [

        "How many rows are there?",

        "How many columns are there?",

        "What columns are available?",

        "Are there any missing values?",

        "Are there duplicate rows?",

        "What is the average salary?",

        "What is the maximum salary?",

        "What is the minimum salary?",

        "What is the median salary?",

        "What are the unique values?",

        "Show me the data",

    ]

    example_cols = st.columns(3)

    if "selected_question" not in st.session_state:

        st.session_state.selected_question = None

    for i, example in enumerate(
        examples
    ):

        if example_cols[
            i % 3
        ].button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):

            st.session_state.selected_question = (
                example
            )

            st.rerun()

    # ========================================================
    # CHAT INPUT
    # ========================================================

    selected_question = (
        st.session_state.selected_question
    )

    question = st.chat_input(
        "Ask something about your data..."
    )

    if question:

        st.session_state.selected_question = None

    elif selected_question:

        question = selected_question

        st.session_state.selected_question = None

    # ========================================================
    # DISPLAY HISTORY
    # ========================================================

    for message in (
        st.session_state.ask_history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message.get("result")
                is not None
            ):

                result = message[
                    "result"
                ]

                if isinstance(
                    result,
                    pd.DataFrame
                ):

                    if not result.empty:

                        st.dataframe(
                            result,
                            use_container_width=True
                        )

            if message.get("sql"):

                with st.expander(
                    "🔍 View SQL"
                ):

                    st.code(
                        message["sql"],
                        language="sql"
                    )

    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if question:

        # ====================================================
        # SAVE USER MESSAGE
        # ====================================================

        st.session_state.ask_history.append(
            {
                "role":
                    "user",

                "content":
                    question,

                "result":
                    None,

                "sql":
                    None
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # ====================================================
        # ASSISTANT
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            if not SQL_ENGINE_AVAILABLE:

                st.error(
                    "sql_engine.py could not be imported."
                )

                st.code(
                    SQL_ENGINE_ERROR
                )

            else:

                with st.spinner(
                    "Analyzing your question..."
                ):

                    try:

                        # ============================================
                        # STEP 1
                        # DIRECT PYTHON ENGINE
                        # ============================================

                        (
                            success,
                            answer,
                            result,
                            sql
                        ) = (
                            answer_question_directly(
                                question,
                                df
                            )
                        )

                        if success:

                            st.markdown(
                                answer
                            )

                            if (
                                result is not None
                                and
                                isinstance(
                                    result,
                                    pd.DataFrame
                                )
                                and
                                not result.empty
                            ):

                                st.dataframe(
                                    result,
                                    use_container_width=True
                                )

                            st.session_state.ask_history.append(
                                {
                                    "role":
                                        "assistant",

                                    "content":
                                        answer,

                                    "result":
                                        result,

                                    "sql":
                                        sql
                                }
                            )

                        else:

                            # ========================================
                            # STEP 2
                            # LOCAL LLM
                            # ========================================

                            if not AI_ENGINE_AVAILABLE:

                                st.warning(
                                    "AI engine is unavailable. "
                                    "Try one of the example questions."
                                )

                            else:

                                st.info(
                                    "This is an advanced question. "
                                    "Using the local AI model..."
                                )

                                if (
                                    st.session_state.llm
                                    is None
                                ):

                                    st.session_state.llm = (
                                        load_local_llm()
                                    )

                                if (
                                    st.session_state.llm
                                    is None
                                ):

                                    st.warning(
                                        """
                                        The local AI model could not
                                        be loaded.

                                        Try one of the supported
                                        direct questions such as:

                                        - Average
                                        - Maximum
                                        - Minimum
                                        - Median
                                        - Missing values
                                        - Duplicates
                                        - Row count
                                        - Column count
                                        """
                                    )

                                else:

                                    # ====================================
                                    # GENERATE SQL
                                    # ====================================

                                    sql = generate_ai_sql(
                                        question,
                                        df,
                                        st.session_state.llm
                                    )

                                    if not sql:

                                        answer = (
                                            "I couldn't generate a "
                                            "reliable query for that "
                                            "question."
                                        )

                                        st.warning(
                                            answer
                                        )

                                        st.session_state.ask_history.append(
                                            {
                                                "role":
                                                    "assistant",

                                                "content":
                                                    answer,

                                                "result":
                                                    None,

                                                "sql":
                                                    None
                                            }
                                        )

                                    else:

                                        # =================================
                                        # VALIDATE
                                        # =================================

                                        valid, message = (
                                            validate_sql(
                                                sql
                                            )
                                        )

                                        if not valid:

                                            st.error(
                                                message
                                            )

                                        else:

                                            # =============================
                                            # EXECUTE
                                            # =============================

                                            result, error = (
                                                execute_query(
                                                    df,
                                                    sql
                                                )
                                            )

                                            if error:

                                                st.error(
                                                    "The generated SQL "
                                                    "could not be executed."
                                                )

                                                with st.expander(
                                                    "Technical Details"
                                                ):

                                                    st.code(
                                                        sql,
                                                        language="sql"
                                                    )

                                                    st.error(
                                                        error
                                                    )

                                            else:

                                                # =========================
                                                # ANSWER
                                                # =========================

                                                if result.empty:

                                                    answer = (
                                                        "The query executed "
                                                        "successfully but "
                                                        "returned no results."
                                                    )

                                                elif (
                                                    result.shape[0] == 1
                                                    and
                                                    result.shape[1] == 1
                                                ):

                                                    value = result.iloc[
                                                        0,
                                                        0
                                                    ]

                                                    if isinstance(
                                                        value,
                                                        (
                                                            float,
                                                            np.floating
                                                        )
                                                    ):

                                                        value = round(
                                                            float(value),
                                                            3
                                                        )

                                                    answer = (
                                                        f"The answer is "
                                                        f"**{value}**."
                                                    )

                                                else:

                                                    answer = (
                                                        f"The query returned "
                                                        f"**{len(result):,} rows**."
                                                    )

                                                st.markdown(
                                                    answer
                                                )

                                                st.dataframe(
                                                    result,
                                                    use_container_width=True
                                                )

                                                with st.expander(
                                                    "🔍 Generated SQL"
                                                ):

                                                    st.code(
                                                        sql,
                                                        language="sql"
                                                    )

                                                st.session_state.ask_history.append(
                                                    {
                                                        "role":
                                                            "assistant",

                                                        "content":
                                                            answer,

                                                        "result":
                                                            result,

                                                        "sql":
                                                            sql
                                                    }
                                                )

                    except Exception as e:

                        st.error(
                            "Something went wrong while "
                            "processing the question."
                        )

                        with st.expander(
                            "Technical Error"
                        ):

                            st.exception(
                                e
                            )

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.session_state.ask_history:

        st.divider()

        if st.button(
            "🗑️ Clear Chat History",
            use_container_width=True
        ):

            st.session_state.ask_history = []

            st.rerun()


# ============================================================
# ML READINESS
# ============================================================


elif page == "🤖 ML Readiness":

    st.markdown(
        '<div class="section-title">🤖 ML Readiness</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Evaluate whether the dataset is ready for machine learning'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # DATASET
    # ========================================================

    st.subheader(
        "📊 Dataset Characteristics"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(df):,}"
    )

    c2.metric(
        "Features",
        f"{len(df.columns):,}"
    )

    c3.metric(
        "Numeric Features",
        profile["numeric"]
    )

    c4.metric(
        "Categorical Features",
        profile["categorical"]
    )

    st.divider()

    # ========================================================
    # TARGET
    # ========================================================

    target = st.selectbox(
        "🎯 Select Target Column",
        ["None"] + list(df.columns)
    )

    if target != "None":

        st.subheader(
            "🎯 Target Analysis"
        )

        target_series = df[
            target
        ]

        if pd.api.types.is_numeric_dtype(
            target_series
        ):

            st.success(
                "Numeric target detected → "
                "Regression may be appropriate."
            )

            st.write(
                target_series.describe()
            )

        else:

            classes = target_series.nunique(
                dropna=True
            )

            st.info(
                f"Categorical target detected "
                f"with {classes} unique classes."
            )

            if classes == 2:

                st.success(
                    "Binary classification problem detected."
                )

            elif classes <= 20:

                st.success(
                    "Multiclass classification may be appropriate."
                )

            else:

                st.warning(
                    "High-cardinality categorical target."
                )

    # ========================================================
    # ML CHECKS
    # ========================================================

    st.subheader(
        "✅ ML Readiness Checklist"
    )

    checks = []

    checks.append(
        (
            "Sufficient rows",
            len(df) >= 100
        )
    )

    checks.append(
        (
            "Missing values under control",
            profile["missing"]
            <=
            len(df) * len(df.columns) * 0.1
        )
    )

    checks.append(
        (
            "Duplicate rows under control",
            profile["duplicates"]
            <=
            len(df) * 0.05
        )
    )

    checks.append(
        (
            "Numeric features available",
            profile["numeric"] > 0
        )
    )

    if target != "None":

        checks.append(
            (
                "Target selected",
                True
            )
        )

    else:

        checks.append(
            (
                "Target selected",
                False
            )
        )

    for name, passed in checks:

        if passed:

            st.success(
                f"✅ {name}"
            )

        else:

            st.warning(
                f"⚠️ {name}"
            )


# ============================================================
# DATA QUALITY
# ============================================================


elif page == "⚙️ Data Quality":

    st.markdown(
        '<div class="section-title">⚙️ Data Quality</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Detailed data-quality diagnostics'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # COLUMN QUALITY TABLE
    # ========================================================

    quality_rows = []

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        unique_count = int(
            df[column].nunique(
                dropna=True
            )
        )

        missing_percentage = (
            missing_count
            /
            max(len(df), 1)
            *
            100
        )

        duplicate_percentage = (
            (
                len(df)
                -
                unique_count
            )
            /
            max(len(df), 1)
            *
            100
        )

        quality_rows.append(
            {
                "Column":
                    column,

                "Data Type":
                    str(
                        df[column].dtype
                    ),

                "Missing":
                    missing_count,

                "Missing %":
                    round(
                        missing_percentage,
                        2
                    ),

                "Unique":
                    unique_count,

                "Unique %":
                    round(
                        unique_count
                        /
                        max(len(df), 1)
                        *
                        100,
                        2
                    ),

                "Memory":
                    f"{df[column].memory_usage(deep=True) / 1024:.1f} KB"
            }
        )

    quality_df = pd.DataFrame(
        quality_rows
    )

    st.dataframe(
        quality_df,
        use_container_width=True,
        height=550
    )

    # ========================================================
    # HIGH MISSING
    # ========================================================

    high_missing = quality_df[
        quality_df["Missing %"] >= 20
    ]

    if not high_missing.empty:

        st.warning(
            "⚠️ Columns with more than 20% missing values:"
        )

        st.dataframe(
            high_missing[
                [
                    "Column",
                    "Missing",
                    "Missing %"
                ]
            ],
            use_container_width=True
        )

    # ========================================================
    # CONSTANT COLUMNS
    # ========================================================

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:

        st.warning(
            "⚠️ Constant columns detected:"
        )

        st.write(
            constant_columns
        )

    else:

        st.success(
            "✅ No constant columns detected."
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    if profile["duplicates"]:

        st.warning(
            f"⚠️ {profile['duplicates']:,} duplicate "
            "rows detected."
        )

        if st.checkbox(
            "Show duplicate rows"
        ):

            st.dataframe(
                df[
                    df.duplicated(
                        keep=False
                    )
                ],
                use_container_width=True
            )

    else:

        st.success(
            "✅ No duplicate rows detected."
        )


# ============================================================
# FOOTER
# ============================================================


st.divider()

st.caption(
    "🤖 AI Data Analyst | Python + Pandas + Statistics + "
    "EDA + ML + NLP/LLM + Data Engineering + Plotly + Streamlit"
)