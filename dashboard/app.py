import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import streamlit as st

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generator import save_generated_data
from src.data_processing import process_and_clean_data
from src.sql_queries import build_database, execute_query, get_completion_rate_by_genre, get_format_pages_per_session
from src.feature_engineering import calculate_behavioral_features, build_ml_feature_set
from src.analysis import run_all_statistical_tests, analyze_format_comparison, analyze_genre_comparison, analyze_time_of_day
from src.modeling import train_and_evaluate_models, explain_single_prediction, get_feature_importances
from src.clustering import perform_reader_clustering
from src.recommendation import generate_behavioral_strategy

# ---------------------------------------------------------
# Page Configuration & Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="Reading Behavior Analytics: E-Books vs Physical",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark aesthetic and cards
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4A90E2, #50E3C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #A0AAB8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222D;
        border: 1px solid #2E3646;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #50E3C2;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #8E9AA8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Initialization
# ---------------------------------------------------------
@st.cache_data
def load_all_pipeline_data():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    db_path = "database/reading_data.db"
    
    if not os.path.exists(os.path.join(processed_dir, "reading_sessions_master.csv")):
        save_generated_data(raw_dir)
        df_readers, df_books, df_sessions, df_master, report = process_and_clean_data(raw_dir, processed_dir)
        build_database(processed_dir, db_path)
    else:
        df_master = pd.read_csv(os.path.join(processed_dir, "reading_sessions_master.csv"))
        df_readers = pd.read_csv(os.path.join(processed_dir, "readers_clean.csv"))
        df_books = pd.read_csv(os.path.join(processed_dir, "books_clean.csv"))
        df_sessions = pd.read_csv(os.path.join(processed_dir, "sessions_clean.csv"))
        
    df_eng, reader_features = calculate_behavioral_features(df_master)
    return df_readers, df_books, df_sessions, df_master, df_eng, reader_features

@st.cache_resource
def load_ml_models(df_eng, reader_features):
    X, y, feature_cols = build_ml_feature_set(df_eng)
    eval_res = train_and_evaluate_models(X, y)
    c_res = perform_reader_clustering(reader_features)
    return eval_res, c_res, feature_cols

df_readers, df_books, df_sessions, df_master, df_eng, reader_features = load_all_pipeline_data()
eval_res, c_res, feature_cols = load_ml_models(df_eng, reader_features)

# ---------------------------------------------------------
# Sidebar & Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">📚 Reading Behavior Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">E-Books vs. Physical Books: Quantitative Research & Behavioral Intelligence Dashboard</div>', unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/isometric-folders/100/open-book.png", width=70)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Module",
    [
        "📊 Executive Overview",
        "📖 Format Comparison (E-Book vs Physical)",
        "👤 Reader & Book Deep-Dive",
        "⏰ Time & Temporal Patterns",
        "🤖 ML Completion Predictor & XAI",
        "🎯 Behavioral Recommendations",
        "🧩 Reader Segmentation & Clustering",
        "💻 SQL Analytical Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Data Source Transparency Notice:**\n"
    "Synthetic reader profiles and session telemetry were generated to simulate a realistic multi-level reading dataset. "
    "Book metadata is also programmatically generated for the prototype."
)

# ---------------------------------------------------------
# 1. Executive Overview Tab
# ---------------------------------------------------------
if menu == "📊 Executive Overview":
    st.header("Executive Overview & Summary Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_readers = len(df_readers)
    total_books = len(df_books)
    total_sessions = len(df_sessions)
    avg_duration = df_sessions['duration_minutes'].mean()
    avg_pages = df_sessions['pages_read'].mean()
    overall_comp = df_sessions['completed'].mean() * 100.0
    
    col1.markdown(f'<div class="metric-card"><div class="metric-val">{total_readers}</div><div class="metric-lbl">Total Readers</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="metric-val">{total_books}</div><div class="metric-lbl">Total Books</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="metric-val">{total_sessions:,}</div><div class="metric-lbl">Total Sessions</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><div class="metric-val">{avg_duration:.1f}m</div><div class="metric-lbl">Avg Duration</div></div>', unsafe_allow_html=True)
    col5.markdown(f'<div class="metric-card"><div class="metric-val">{avg_pages:.1f}</div><div class="metric-lbl">Avg Pages/Sess</div></div>', unsafe_allow_html=True)
    col6.markdown(f'<div class="metric-card"><div class="metric-val">{overall_comp:.1f}%</div><div class="metric-lbl">Completion Rate</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.subheader("Reading Sessions by Format & Device")
        fig_fmt = px.pie(
            df_sessions, names='format', title="Format Distribution (Physical vs E-Book)",
            hole=0.4, color_discrete_sequence=['#4A90E2', '#50E3C2']
        )
        st.plotly_chart(fig_fmt, use_container_width=True)
        
    with c_right:
        st.subheader("Reading Sessions Across Device Types")
        fig_dev = px.bar(
            df_sessions.groupby('device_type').size().reset_index(name='count'),
            x='device_type', y='count', color='device_type',
            title="Sessions Count per Device Type",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_dev, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Key Findings Summary")
    st.markdown("""
    - **Speed Variance:** E-Book readers show slightly higher average pages per minute (PPM) compared to physical book readers, consistent with digital scanning habits.
    - **Completion Dynamics:** Physical book sessions show a slight edge in overall book completion rate, driven by higher focus window durations.
    - **Temporal Peaks:** Reading activity peaks in the evening hours (18:00 - 22:00) and weekend periods.
    """)

# ---------------------------------------------------------
# 2. Format Comparison Tab
# ---------------------------------------------------------
elif menu == "📖 Format Comparison (E-Book vs Physical)":
    st.header("Format Comparison: E-Books vs Physical Books")
    
    df_fmt_summary = analyze_format_comparison(df_master)
    st.dataframe(df_fmt_summary, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reading Speed (Pages / Min) Distribution")
        fig_box_speed = px.box(
            df_master, x='format', y='pages_per_minute', color='format',
            points="outliers", title="Reading Speed Comparison (PPM)",
            color_discrete_map={'Ebook': '#50E3C2', 'Physical': '#4A90E2'}
        )
        st.plotly_chart(fig_box_speed, use_container_width=True)
        
    with col2:
        st.subheader("Session Duration (Minutes) Distribution")
        fig_box_dur = px.box(
            df_master, x='format', y='duration_minutes', color='format',
            points="outliers", title="Session Duration Comparison (Min)",
            color_discrete_map={'Ebook': '#50E3C2', 'Physical': '#4A90E2'}
        )
        st.plotly_chart(fig_box_dur, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Statistical Hypothesis Testing Results")
    stat_res = run_all_statistical_tests(df_master)
    
    s_test = stat_res['speed_by_format']
    c_test = stat_res['completion_by_format']
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### Hypothesis 1: Reading Speed Difference")
        st.markdown(f"- **E-Book Mean Speed:** {s_test['group_1_mean']} PPM (95% CI: {s_test['group_1_ci']})")
        st.markdown(f"- **Physical Mean Speed:** {s_test['group_2_mean']} PPM (95% CI: {s_test['group_2_ci']})")
        st.markdown(f"- **Welch's t-stat:** `{s_test['t_statistic']}`, **p-value:** `{s_test['p_value_ttest']:.4e}`")
        st.markdown(f"- **Cohen's d Effect Size:** `{s_test['cohens_d']}`")
        st.success(s_test['practical_interpretation'])
        
    with col_b:
        st.markdown(f"#### Hypothesis 2: Completion Rate Independence")
        st.markdown(f"- **Chi-Square Stat:** `{c_test['chi2_statistic']}`, **p-value:** `{c_test['p_value']:.4e}`")
        st.markdown(f"- **Cramér's V Effect Size:** `{c_test['cramers_v']}`")
        st.info(c_test['practical_interpretation'])

# ---------------------------------------------------------
# 3. Reader & Book Deep-Dive Tab
# ---------------------------------------------------------
elif menu == "👤 Reader & Book Deep-Dive":
    st.header("Reader & Book Behavioral Deep-Dive")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    sel_genre = col_f1.selectbox("Filter by Genre", ["All"] + list(df_books['genre'].unique()))
    sel_fmt = col_f2.selectbox("Filter by Format", ["All", "Physical", "Ebook"])
    sel_exp = col_f3.selectbox("Filter by Experience Tier", ["All"] + list(df_readers['reading_experience'].unique()))
    
    df_filtered = df_master.copy()
    if sel_genre != "All":
        df_filtered = df_filtered[df_filtered['genre'] == sel_genre]
    if sel_fmt != "All":
        df_filtered = df_filtered[df_filtered['format'] == sel_fmt]
    if sel_exp != "All":
        df_filtered = df_filtered[df_filtered['reading_experience'] == sel_exp]
        
    st.subheader(f"Filtered Sessions Dataset ({len(df_filtered)} records)")
    st.dataframe(
        df_filtered[['session_id', 'reader_id', 'title', 'genre', 'format', 'duration_minutes', 'pages_read', 'pages_per_minute', 'completed']].head(100),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("Scatter Plot: Session Duration vs Pages Read")
    fig_scatter = px.scatter(
        df_filtered, x='duration_minutes', y='pages_read', color='format',
        size='percentage_completed', hover_data=['title', 'reader_id'],
        title="Session Duration vs Pages Read (Sized by Completion %)",
        color_discrete_map={'Ebook': '#50E3C2', 'Physical': '#4A90E2'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# 4. Time Analysis Tab
# ---------------------------------------------------------
elif menu == "⏰ Time & Temporal Patterns":
    st.header("Time-of-Day & Temporal Reading Patterns")
    
    df_tod = analyze_time_of_day(df_master)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sessions Count by Time of Day")
        fig_tod = px.bar(
            df_tod, x='time_of_day', y='total_sessions', color='time_of_day',
            title="Session Volume Across Time Slots",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_tod, use_container_width=True)
        
    with c2:
        st.subheader("Completion Rate by Time of Day")
        fig_tod_comp = px.bar(
            df_tod, x='time_of_day', y='completion_rate', color='time_of_day',
            title="Book Completion Rate by Time Slot",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_tod_comp, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Day of Week Activity Heatmap")
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_df = df_master.groupby(['day_of_week', 'time_of_day']).size().unstack(fill_value=0)
    dow_df = dow_df.reindex(dow_order)
    
    fig_heat = px.imshow(
        dow_df, labels=dict(x="Time of Day", y="Day of Week", color="Sessions"),
        x=dow_df.columns, y=dow_df.index, title="Reading Session Heatmap (Day vs Time)",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------
# 5. ML Completion Predictor Tab
# ---------------------------------------------------------
elif menu == "🤖 ML Completion Predictor & XAI":
    st.header("Predicting Book Completion & Explainable AI")
    
    st.markdown("""
    Use the trained Machine Learning classifier (Random Forest / Logistic Regression) to estimate whether a reader will complete a book based on session behavioral features.
    """)
    
    best_name = eval_res['best_model_name']
    st.success(f"Active ML Model: **{best_name}** | Test F1-Score: **{eval_res['results_summary'].loc[best_name, 'f1_score']}** | ROC-AUC: **{eval_res['results_summary'].loc[best_name, 'roc_auc']}**")
    
    st.dataframe(eval_res['results_summary'], use_container_width=True)
    
    st.markdown("---")
    st.subheader("Interactive Prediction & XAI Input")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    in_fmt = col_in1.selectbox("Format", ["Ebook", "Physical"])
    in_genre = col_in2.selectbox("Genre", list(df_books['genre'].unique()))
    in_pages = col_in3.number_input("Book Page Count", min_value=50, max_value=1200, value=320)
    
    col_in4, col_in5, col_in6 = st.columns(3)
    in_dur = col_in4.slider("Average Session Duration (Min)", 5, 120, 35)
    in_freq = col_in5.slider("Sessions Per Week", 0.5, 14.0, 3.5, step=0.5)
    in_gap = col_in6.slider("Average Gap Between Sessions (Days)", 0.0, 14.0, 2.0, step=0.5)
    
    if st.button("Predict Completion Probability"):
        # Construct dummy vector matching feature_cols
        input_dict = {col: 0 for col in feature_cols}
        
        input_dict['page_count'] = in_pages
        input_dict['duration_minutes'] = in_dur
        input_dict['pages_read'] = int(in_dur * 0.6)
        input_dict['pages_per_minute'] = 0.6
        input_dict['reader_prior_avg_duration'] = in_dur
        input_dict['reader_prior_avg_pages'] = int(in_dur * 0.6)
        input_dict['reader_prior_avg_ppm'] = 0.6
        input_dict['reader_prior_completion_rate'] = 0.5
        input_dict['reader_prior_sessions_count'] = 3
        input_dict['reader_days_since_last_session'] = in_gap
        input_dict['reading_frequency_per_week'] = in_freq
        input_dict['average_gap_between_sessions'] = in_gap
        
        # Category dummies
        fmt_col = f"format_{in_fmt}"
        if fmt_col in input_dict:
            input_dict[fmt_col] = 1
            
        genre_col = f"genre_clean_{in_genre}" if f"genre_clean_{in_genre}" in input_dict else f"genre_{in_genre}"
        if genre_col in input_dict:
            input_dict[genre_col] = 1
            
        input_df = pd.DataFrame([input_dict])[feature_cols]
        
        explanation = explain_single_prediction(
            eval_res['best_model'], eval_res['scaler'], input_df, feature_cols
        )
        
        prob_pct = explanation['completion_pct']
        st.markdown(f"### Estimated Completion Probability: `{prob_pct}%`")
        st.progress(prob_pct / 100.0)
        
        c_x1, c_x2 = st.columns(2)
        with c_x1:
            st.success("##### 🟢 Top Factors Increasing Completion")
            for f in explanation['increasing_factors']:
                st.markdown(f"- {f}")
        with c_x2:
            st.warning("##### 🔴 Top Factors Decreasing Completion")
            for f in explanation['decreasing_factors']:
                st.markdown(f"- {f}")

# ---------------------------------------------------------
# 6. Behavioral Recommendations Tab
# ---------------------------------------------------------
elif menu == "🎯 Behavioral Recommendations":
    st.header("Personalized Behavioral Strategy Recommendations")
    
    st.markdown("""
    This behavioral intelligence engine converts a reader's historical session patterns into customized, actionable reading strategies to maximize completion.
    """)
    
    sel_reader_id = st.selectbox("Select Reader ID", df_readers['reader_id'].unique()[:50])
    
    r_row = reader_features[reader_features['reader_id'] == sel_reader_id].iloc[0].to_dict()
    
    strat = generate_behavioral_strategy(r_row)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.info(f"**Recommended Session Window:**\n{strat['recommended_session_window']}")
    col_s2.success(f"**Target Weekly Frequency:**\n{strat['target_sessions_per_week']} sessions / week")
    col_s3.warning(f"**Optimal Time Slot:**\n{strat['optimal_time_of_day']}")
    
    st.markdown("---")
    st.subheader("Custom Behavioral Strategy Insights")
    for insight in strat['behavioral_insights']:
        st.markdown(f"- {insight}")

# ---------------------------------------------------------
# 7. Reader Clustering Tab
# ---------------------------------------------------------
elif menu == "🧩 Reader Segmentation & Clustering":
    st.header("Unsupervised Reader Segmentation (K-Means & PCA)")
    
    st.markdown(f"K-Means Silhouette Score: **{c_res['silhouette_kmeans']}** | Derived 4 Archetypes")
    
    df_clustered = c_res['reader_clustered_df']
    
    fig_pca = px.scatter(
        df_clustered, x='pca_x', y='pca_y', color='segment_name',
        hover_data=['reader_id', 'reader_unique_books', 'reader_avg_session_duration'],
        title="2D PCA Projection of Reader Clusters",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    st.subheader("Cluster Centroids & Behavioral Characteristics")
    st.dataframe(c_res['cluster_profiles'], use_container_width=True)

# ---------------------------------------------------------
# 8. SQL Analytical Explorer Tab
# ---------------------------------------------------------
elif menu == "💻 SQL Analytical Explorer":
    st.header("SQLite Database & SQL Explorer")
    
    st.markdown("Execute SQL queries directly against `database/reading_data.db`.")
    
    query_option = st.selectbox(
        "Choose Pre-configured Analytical Query",
        [
            "1. Completion Rate by Genre",
            "2. Format Average Pages per Session",
            "3. Top 10 Readers by Completion Rate",
            "4. Reading Speed by Genre",
            "5. Monthly Activity Trends",
            "6. Custom SQL Query"
        ]
    )
    
    if query_option == "1. Completion Rate by Genre":
        sql = "SELECT b.genre, COUNT(s.session_id) AS total_sessions, ROUND(AVG(s.completed)*100.0, 2) AS completion_pct FROM sessions s JOIN books b ON s.book_id = b.book_id GROUP BY b.genre ORDER BY completion_pct DESC;"
    elif query_option == "2. Format Average Pages per Session":
        sql = "SELECT format, COUNT(session_id) AS sessions_count, ROUND(AVG(pages_read), 2) AS avg_pages, ROUND(AVG(duration_minutes), 2) AS avg_duration_min FROM sessions GROUP BY format;"
    elif query_option == "3. Top 10 Readers by Completion Rate":
        sql = "SELECT r.reader_id, r.age_group, r.reading_experience, COUNT(s.session_id) AS total_sessions, ROUND(AVG(s.completed)*100.0, 2) AS comp_pct FROM readers r JOIN sessions s ON r.reader_id = s.reader_id GROUP BY r.reader_id HAVING total_sessions >= 5 ORDER BY comp_pct DESC LIMIT 10;"
    elif query_option == "4. Reading Speed by Genre":
        sql = "SELECT b.genre, ROUND(AVG(s.pages_per_minute), 3) AS avg_speed_ppm FROM sessions s JOIN books b ON s.book_id = b.book_id GROUP BY b.genre ORDER BY avg_speed_ppm DESC;"
    elif query_option == "5. Monthly Activity Trends":
        sql = "SELECT STRFTIME('%Y-%m', session_date) AS month, COUNT(session_id) AS total_sessions, SUM(pages_read) AS pages_sum FROM sessions GROUP BY month ORDER BY month;"
    else:
        sql = st.text_area("Write Custom SQL Query", "SELECT * FROM sessions LIMIT 10;")
        
    st.code(sql, language="sql")
    
    if st.button("Run SQL Query"):
        try:
            res_df = execute_query(sql)
            st.dataframe(res_df, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")
