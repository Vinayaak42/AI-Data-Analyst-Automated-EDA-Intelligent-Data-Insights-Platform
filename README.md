# 🤖 AI Data Analyst – Automated EDA & Intelligent Data Insights Platform

> An AI-powered automated data analysis platform that transforms raw CSV, Excel, JSON, and Parquet files into meaningful insights using Python, Statistics, EDA, Machine Learning, NLP/LLMs, Data Engineering, SQL, and interactive visualizations.

---

## 📌 Project Overview

**AI Data Analyst** is an intelligent data analysis application built with **Python and Streamlit**.

The platform allows users to upload a dataset and automatically performs:

- Data loading and validation
- Basic data cleaning
- Exploratory Data Analysis (EDA)
- Statistical analysis
- Missing-value analysis
- Duplicate detection
- Outlier detection
- Correlation analysis
- Pattern discovery
- Interactive visualization
- AI-powered insight generation
- Natural-language data querying
- SQL generation and execution
- Machine Learning readiness analysis
- Data-quality assessment

Instead of manually writing Python code for every dataset, users can upload their data and let the application perform the initial stages of the data-analysis workflow automatically.

---

# 🎯 Project Objective

The main objective of this project is to build an **AI-assisted data analyst that automates the initial stages of a real-world data science workflow**.

Traditional data analysis often requires analysts to manually:

1. Load the dataset
2. Inspect the schema
3. Clean the data
4. Calculate statistics
5. Identify missing values
6. Detect outliers
7. Analyze correlations
8. Create visualizations
9. Discover patterns
10. Generate business insights

This project combines these steps into a single interactive platform.

---

# 🚀 Key Features

## 1. 📂 Multi-Format Dataset Upload

Users can upload:

- CSV
- Excel
- XLS
- JSON
- Parquet

The application automatically detects the file type and loads the dataset.

---

## 2. 🧹 Automatic Data Cleaning

The system performs basic preprocessing such as:

- Removing completely empty rows
- Removing completely empty columns
- Cleaning column names
- Detecting data types
- Identifying missing values
- Detecting duplicate rows

---

## 3. 📊 Automatic Exploratory Data Analysis

The application automatically analyzes the structure of the uploaded dataset.

### Dataset Overview

Displays:

- Number of rows
- Number of columns
- Numeric columns
- Categorical columns
- Datetime columns
- Missing cells
- Duplicate rows
- Memory usage

---

## 4. 📈 Statistical Analysis

For numeric columns, the system calculates:

- Mean
- Median
- Standard deviation
- Minimum
- Maximum
- Quartiles
- Count
- Missing values
- Unique values

Example:

```text
Salary

Mean       : 52,430
Median     : 49,500
Std        : 12,430
Minimum    : 21,000
Maximum    : 98,000
```

---

## 5. 🚨 Automatic Outlier Detection

The application uses the **Interquartile Range (IQR)** method to identify potential outliers.

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

The system reports:

- Number of outliers
- Outlier percentage
- Lower bound
- Upper bound

---

## 6. ⚠️ Missing Value Analysis

The application automatically identifies:

- Columns containing missing values
- Number of missing values
- Missing-value percentage

Example:

```text
Column       Missing      Missing %
-----------------------------------
Age             12           2.4%
Salary           4           0.8%
Department       0           0.0%
```

---

## 7. ♻️ Duplicate Detection

The application automatically checks for duplicate rows.

Users can also view duplicate records directly.

---

## 8. 🔗 Correlation Analysis

For datasets containing multiple numeric columns, the system generates a correlation matrix.

This helps identify relationships such as:

```text
Experience ↔ Salary
Age        ↔ Income
Hours      ↔ Performance
```

Interactive correlation heatmaps are generated using Plotly.

---

## 9. 📊 Interactive Visualizations

The platform supports:

### Histogram

Used for understanding numerical distributions.

### Box Plot

Used for:

- Distribution
- Median
- Quartiles
- Outliers

### Scatter Plot

Used to analyze relationships between numeric variables.

### Bar Chart

Used for categorical analysis.

### Line Chart

Used for trends and time-series-like data.

### Pie Chart

Used for category proportions.

### Correlation Heatmap

Used to analyze relationships between numerical variables.

---

# 10. 💡 AI-Powered Insights

The AI Insights module combines verified Python-based analysis with a local AI model to generate meaningful interpretations.

Architecture:

```text
Python Statistical Analysis
             ↓
Verified Dataset Findings
             ↓
Local AI Model
             ↓
Natural Language Interpretation
             ↓
AI Insights
```

The system first calculates numerical findings using Python and then uses the AI layer to interpret those findings.

This helps reduce the risk of an LLM inventing numerical values.

---

# 11. 💬 Ask Your Data

The platform provides a natural-language interface for querying datasets.

Users can ask questions such as:

```text
How many rows are there?

What is the average salary?

What is the maximum salary?

What is the minimum age?

What is the median income?

Are there missing values?

Are there duplicate rows?

What columns are available?

Show me the data.
```

The system first attempts to answer common analytical questions directly using Python.

For more complex questions, the application can use a local LLM to generate SQL.

Architecture:

```text
User Question
      ↓
Question Understanding
      ↓
Direct Python Analysis
      ↓
If Required
      ↓
Local LLM
      ↓
SQL Generation
      ↓
SQL Validation
      ↓
DuckDB
      ↓
Result
      ↓
User
```

---

# 12. 🤖 Local LLM Integration

The project is designed to support **local Hugging Face models**.

The local AI layer can be used for:

- Natural-language understanding
- AI insight generation
- SQL generation
- Data-question interpretation

This allows the project to experiment with AI-powered data analysis without depending entirely on paid external APIs.

---

# 13. 🗄️ SQL-Based Data Analysis

The project can use **DuckDB** for analytical SQL queries over Pandas DataFrames.

Example:

```sql
SELECT AVG(Salary)
FROM dataset;
```

Another example:

```sql
SELECT Department, AVG(Salary)
FROM dataset
GROUP BY Department;
```

This creates a bridge between:

```text
Natural Language
        ↓
AI
        ↓
SQL
        ↓
Data
        ↓
Result
```

---

# 14. 🤖 Machine Learning Readiness

The application evaluates whether a dataset is suitable for machine learning.

It checks:

- Number of observations
- Number of features
- Numeric features
- Categorical features
- Missing values
- Duplicate records
- Target variable
- Target type
- Number of classes

It can identify potential situations such as:

```text
⚠ High missing-value percentage

⚠ Too many duplicate records

⚠ Constant features

⚠ High-cardinality target

✓ Numeric target → possible regression

✓ Binary target → possible classification
```

---

# 15. ⚙️ Data Quality Analysis

The Data Quality module provides column-level diagnostics.

For every column, the application analyzes:

- Data type
- Missing values
- Missing percentage
- Unique values
- Unique percentage
- Memory usage

It also detects:

- Constant columns
- High missing-value columns
- Duplicate rows

---

# 🏗️ Application Architecture

```text
                         ┌─────────────────────┐
                         │        User         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Streamlit       │
                         │     Interface       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    File Upload      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Data Loading       │
                         │ Pandas / File I/O   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Data Cleaning      │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
          │     EDA     │    │ Statistics  │    │ Data Quality│
          └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Pattern Discovery   │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
          ┌─────────────────┐               ┌─────────────────┐
          │ Visualization   │               │   AI Insights   │
          └─────────────────┘               └────────┬────────┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │ Local Hugging   │
                                             │ Face LLM        │
                                             └─────────────────┘

                         Ask Your Data
                               │
                               ▼
                     ┌────────────────────┐
                     │ Natural Language   │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │ Python Direct      │
                     │ Analysis           │
                     └─────────┬──────────┘
                               │
                       Complex Question?
                         /          \
                       No            Yes
                       │              │
                       ▼              ▼
                    Result       Local LLM
                                      │
                                      ▼
                                 SQL Generation
                                      │
                                      ▼
                                   DuckDB
                                      │
                                      ▼
                                   Result
```

---

# 📁 Project Structure

```text
AI_Data_Analyst/
│
├── streamlit_app.py
├── ai_engine.py
├── sql_engine.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
└── notebooks/
    └── .gitkeep
```

---

# 🛠️ Technology Stack

## Programming

- Python

## Data Analysis

- Pandas
- NumPy

## Statistics

- Descriptive Statistics
- IQR
- Correlation Analysis
- Distribution Analysis

## Visualization

- Plotly
- Streamlit

## Machine Learning

- Scikit-learn
- ML readiness analysis

## NLP / LLM

- Hugging Face Transformers
- Local LLMs
- Natural Language Processing
- Prompt Engineering

## SQL / Data Engineering

- DuckDB
- SQL
- Pandas DataFrames
- Data validation

## Application

- Streamlit

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Data-Analyst.git
```

Move into the project:

```bash
cd AI-Data-Analyst
```

---

## 2. Create Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## 4. Run the Application

```bash
python -m streamlit run streamlit_app.py
```

The application will open in your browser.

---

# 📊 Example Workflow

```text
1. Upload Dataset
        ↓
2. Dataset Automatically Loaded
        ↓
3. Data Profile Created
        ↓
4. EDA Performed
        ↓
5. Missing Values Detected
        ↓
6. Duplicates Detected
        ↓
7. Outliers Detected
        ↓
8. Correlations Calculated
        ↓
9. Charts Generated
        ↓
10. AI Insights Generated
        ↓
11. Ask Questions Using Natural Language
        ↓
12. ML Readiness Evaluated
```

---

# 🧪 Example Questions

## Basic Questions

```text
How many rows are there?
```

```text
How many columns are there?
```

```text
What columns are available?
```

## Statistical Questions

```text
What is the average salary?
```

```text
What is the median age?
```

```text
What is the maximum income?
```

```text
What is the minimum price?
```

## Data Quality Questions

```text
Are there missing values?
```

```text
Are there duplicate rows?
```

## Analytical Questions

```text
Which department has the highest salary?
```

```text
What are the most common categories?
```

```text
Show me the top 10 records.
```

---

# 🔐 Privacy & Local AI

The project follows a **local-first AI architecture**.

The goal is to allow datasets to be analyzed locally without requiring users to upload business datasets to external AI services.

Local AI can be used for:

- Insight generation
- Natural-language understanding
- SQL generation

This architecture is useful for experimenting with private datasets in a local environment.

---

# ⚡ Performance Considerations

Large datasets and local LLMs may require additional:

- RAM
- CPU
- Storage

For systems with limited RAM, lightweight local models are recommended.

Future versions can introduce:

- Dataset sampling
- Chunked processing
- Lazy loading
- DuckDB-native processing
- Caching
- Background processing
- Vectorized computations

---

# 🔮 Future Roadmap

## Phase 1 — Core Data Analysis

- [x] CSV upload
- [x] Excel upload
- [x] JSON upload
- [x] Parquet upload
- [x] Data profiling
- [x] EDA
- [x] Statistics
- [x] Missing-value analysis
- [x] Duplicate detection
- [x] Outlier detection

## Phase 2 — Visualization

- [x] Histogram
- [x] Box plot
- [x] Scatter plot
- [x] Bar chart
- [x] Line chart
- [x] Pie chart
- [x] Correlation heatmap
- [ ] Automatic chart recommendation
- [ ] Advanced dashboards

## Phase 3 — AI Analyst

- [x] AI-generated insights
- [x] Natural-language data questions
- [x] Local LLM integration
- [x] SQL generation
- [x] SQL execution
- [ ] Conversational memory
- [ ] Multi-turn data analysis
- [ ] AI-generated business recommendations

## Phase 4 — Machine Learning

- [x] ML readiness analysis
- [ ] Automatic problem-type detection
- [ ] Automatic preprocessing
- [ ] Automatic feature engineering
- [ ] Model selection
- [ ] Automated model training
- [ ] Hyperparameter tuning
- [ ] Model evaluation
- [ ] Explainable AI
- [ ] SHAP analysis

## Phase 5 — Advanced Analytics

Planned features:

- Time-series analysis
- Trend detection
- Seasonality detection
- Forecasting
- Anomaly detection
- Clustering
- Customer segmentation
- Feature importance
- Statistical hypothesis testing
- A/B testing

## Phase 6 — AI Data Scientist

Future versions aim to support:

```text
Upload Dataset
       ↓
AI Understands Dataset
       ↓
Automatic EDA
       ↓
Automatic Visualization
       ↓
Pattern Detection
       ↓
Statistical Testing
       ↓
Feature Engineering
       ↓
ML Model Selection
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Explainability
       ↓
Business Recommendations
       ↓
Automated Report
```

---

# 📈 Example Use Cases

## Business Analytics

Analyze:

- Sales
- Revenue
- Customers
- Products
- Transactions

## HR Analytics

Analyze:

- Employee turnover
- Salary
- Performance
- Attendance
- Department statistics

## Finance

Analyze:

- Expenses
- Revenue
- Profit
- Transactions
- Financial trends

## Marketing

Analyze:

- Campaign performance
- Customer segmentation
- Conversion rates
- Marketing channels

## Manufacturing

Analyze:

- Production data
- Quality metrics
- Defect rates
- Machine performance
- Process variations

## Data Science

Use the platform for:

- Dataset exploration
- Feature discovery
- Data quality assessment
- ML preprocessing preparation
- Model development preparation

---

# 🧠 What Makes This Project Different?

Traditional EDA tools primarily display statistics and charts.

This project combines multiple parts of the modern data-science workflow:

```text
Python
   +
Statistics
   +
EDA
   +
Data Engineering
   +
Visualization
   +
Machine Learning
   +
NLP
   +
Local LLM
   +
SQL
   +
Streamlit
```

The result is an **AI-assisted data analysis platform** rather than a simple visualization dashboard.

---

# 🎓 Skills Demonstrated

This project demonstrates practical knowledge of:

- Python
- Pandas
- NumPy
- Statistics
- Exploratory Data Analysis
- Data Cleaning
- Data Quality
- Data Visualization
- Plotly
- Streamlit
- SQL
- DuckDB
- Machine Learning concepts
- NLP
- Large Language Models
- Hugging Face
- Prompt Engineering
- AI Automation
- Data Engineering
- Software Architecture
- Git & GitHub

---

# 📌 Resume Project Description

### AI Data Analyst — Automated EDA & AI Insight Platform

**Python | Pandas | NumPy | Statistics | Plotly | Streamlit | DuckDB | Hugging Face | NLP | Machine Learning**

Built an AI-powered data analysis platform that automatically processes CSV, Excel, JSON, and Parquet datasets, performs exploratory data analysis, statistical profiling, missing-value and outlier detection, correlation analysis, and interactive visualization. Integrated local Hugging Face LLMs for automated insight generation and natural-language data querying, with an NL-to-SQL pipeline using DuckDB for analytical queries. Added ML-readiness and data-quality assessment modules to automate the initial stages of the data-science workflow.

---

# 🏆 Project Highlights

```text
✓ Automated EDA
✓ Automated statistical profiling
✓ Multi-format dataset support
✓ Interactive visualizations
✓ Missing-value analysis
✓ Duplicate detection
✓ IQR-based outlier detection
✓ Correlation analysis
✓ AI-generated insights
✓ Natural-language data querying
✓ Local Hugging Face LLM support
✓ Natural Language → SQL
✓ DuckDB analytical execution
✓ ML readiness analysis
✓ Data quality diagnostics
✓ Streamlit interactive interface
```

---

# 👨‍💻 Author

**Vinayak Kesti**

Data Science | Data Analytics | Machine Learning | AI

---

# ⭐ Contributing

Contributions are welcome.

You can contribute by:

1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Testing the application
5. Creating a pull request

Example:

```bash
git checkout -b feature/new-analysis
```

```bash
git add .
```

```bash
git commit -m "Add new analysis feature"
```

```bash
git push origin feature/new-analysis
```

---

# 📄 License

This project is intended for educational, portfolio, and research purposes.

A specific open-source license such as MIT can be added when publishing the repository.

---

# 🚀 Vision

The long-term goal of **AI Data Analyst** is to evolve from an automated EDA application into a complete **AI Data Scientist platform** capable of understanding datasets, discovering patterns, generating insights, building machine-learning models, explaining results, and producing actionable recommendations with minimal manual intervention.

```text
                 ┌──────────────────────┐
                 │      RAW DATA        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   AI DATA ANALYST    │
                 └──────────┬───────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
          EDA          VISUALIZATION       AI
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                     PATTERN DISCOVERY
                            │
                            ▼
                    STATISTICAL ANALYSIS
                            │
                            ▼
                     MACHINE LEARNING
                            │
                            ▼
                    BUSINESS INSIGHTS
                            │
                            ▼
                  🚀 AI DATA SCIENTIST
```

---

**Built with Python 🐍 | Streamlit ⚡ | Pandas 📊 | Hugging Face 🤗 | DuckDB 🦆 | Plotly 📈**
