# Reading Behavior Analytics: E-Books vs Physical Books 📚

A portfolio-level Data Science & Machine Learning project investigating quantitative differences in reading behavior between digital e-books and physical printed books using multi-level session logs, statistical hypothesis testing, explainable AI, unsupervised clustering, SQLite queries, and an interactive Streamlit dashboard.

---

## 📌 Key Findings

```text
Key Findings Summary
──────────────────────────────────────────────────────────────────────────
📱 E-books were associated with higher median pages-per-minute (PPM) rates
📖 Physical book sessions were associated with longer focus durations
🌙 Evening (18:00 - 22:00) emerged as the most active reading period
📚 Higher reading frequency was associated with higher book completion rates
```

---

## 🌟 Key Highlights & Features

- **End-to-End Data Pipeline:** From raw session telemetry and survey profiles (`data/raw/`) to cleaned schemas, outlier management, and SQLite database storage (`database/reading_data.db`).
- **Leak-Free ML Pipeline:** Predicts book completion (`completed = 0/1`) strictly using chronological expanding prior features, fitted with scaling and baseline controls on training splits.
- **Statistical Hypothesis Testing:** Rigorous Welch's $t$-tests, Mann-Whitney U tests, Chi-Square independence tests, and ANOVA with effect sizes (Cohen's $d$, Cramér's $V$) and 95% confidence intervals.
- **Explainable AI (XAI):** Feature contribution breakdown explaining why specific reader habits increase or decrease completion probability.
- **Reader Segmentation:** K-Means & Hierarchical clustering deriving 4 reader archetypes (*Daily Reader*, *Weekend Reader*, *Speed Reader*, *Abandoner*) mapped via 2D/3D PCA.
- **Personalized Behavioral Recommendation Engine:** Suggests custom reading strategies (session window, weekly target, optimal time slot).
- **Multi-Page Interactive Streamlit Dashboard:** 8 interactive tabs powered by Plotly charts and live SQLite query runner.
- **Automated Pytest Suite:** 16+ automated tests verifying data processing, leak-free feature engineering, SQL queries, and model inference.

---

## 🏗️ Project Architecture

```text
reading-behavior-analytics/
│
├── data/
│   ├── raw/                  # Synthesized raw reader, book, and session CSVs
│   └── processed/            # Cleaned and master analytical datasets
│
├── database/
│   ├── reading_data.db       # SQLite database storing normalized tables
│   └── schema.sql            # Table DDL definitions
│
├── notebooks/                # Runnable analytical Python scripts
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_statistical_analysis.py
│   ├── 04_machine_learning.py
│   └── 05_clustering.py
│
├── src/                      # Production source packages
│   ├── __init__.py
│   ├── data_generator.py     # Data synthesizer with multi-level relationship
│   ├── data_processing.py    # Preprocessing, schema validation & cleaning
│   ├── feature_engineering.py# Chronological expanding prior feature engineering
│   ├── analysis.py           # Hypothesis testing & statistical engines
│   ├── sql_queries.py        # SQLite query executor and queries
│   ├── modeling.py           # ML classifiers, baseline controls & XAI
│   ├── clustering.py         # K-Means, Hierarchical Clustering & PCA
│   ├── recommendation.py     # Personalized behavioral strategy generator
│   └── main.py               # Full pipeline orchestrator
│
├── dashboard/
│   └── app.py                # Multi-tab Streamlit dashboard
│
├── models/                   # Saved ML models, scalers, and clusterers
│
├── tests/                    # Automated pytest suite (16 tests)
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   ├── test_sql_queries.py
│   └── test_modeling.py
│
├── docs/
│   └── research_report.md    # 16-chapter Data Science Research Report
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Model Evaluation Summary

Evaluating classifiers on a 20% holdout test dataset (2,657 sessions) using **leak-free prior features**:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 0.8957 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **Logistic Regression** | 0.5438 | 0.1215 | 0.5415 | 0.1984 | 0.5463 |
| **Random Forest** | **0.8957** | 0.0000 | 0.0000 | 0.0000 | 0.4864 |
| **Gradient Boosting** | 0.8950 | 0.0000 | 0.0000 | 0.0000 | **0.5753** |
| **XGBoost** | 0.8920 | 0.1429 | 0.0072 | 0.0137 | 0.5454 |

> [!NOTE]
> Predicting session-level completion without target peeking reflects real-world session outcome imbalance (~10% completion rate). Gradient Boosting achieves modest predictive lift over the Majority Class Baseline.

---

## 🚀 Quickstart & How to Run

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Full End-to-End Pipeline
Execute data generation, cleaning, database creation, statistical testing, ML training, and clustering in one command:
```bash
python -m src.main
```

### 3. Launch Interactive Streamlit Dashboard
Launch the dashboard web application:
```bash
streamlit run dashboard/app.py
```

### 4. Run Analytical Notebook Scripts
Execute any of the individual modular scripts:
```bash
python notebooks/01_data_cleaning.py
python notebooks/02_eda.py
python notebooks/03_statistical_analysis.py
python notebooks/04_machine_learning.py
python notebooks/05_clustering.py
```

### 5. Run Automated Tests (16+ Tests)
Execute the pytest suite:
```bash
pytest tests/ -v
```

---

## 📜 Research Report & Transparency
- Read the full 16-chapter data science report in [docs/research_report.md](file:///c:/Users/SMART%20TECH/Desktop/reader/docs/research_report.md).
- **Synthetic Data Disclaimer:** Synthetic reader profiles and session telemetry were generated to simulate a realistic multi-level reading dataset. Book metadata is also programmatically generated for the prototype.
