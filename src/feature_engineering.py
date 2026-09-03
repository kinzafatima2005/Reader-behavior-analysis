import pandas as pd
import numpy as np

def calculate_behavioral_features(df_master):
    """
    Computes both:
    1. Leak-free expanding chronological prior features for session-level ML prediction.
    2. Global reader-level profile features for unsupervised reader clustering.
    """
    df = df_master.copy()
    
    # Ensure session_date is datetime
    if 'session_date' in df.columns:
        df['session_date'] = pd.to_datetime(df['session_date'])
        
    # 1. Basic Session Speed Metrics
    df['pages_per_minute'] = np.where(df['duration_minutes'] > 0, df['pages_read'] / df['duration_minutes'], 0)
    df['minutes_per_page'] = np.where(df['pages_read'] > 0, df['duration_minutes'] / df['pages_read'], np.nan)
    
    # 2. Chronological Sorting for Leak-Free Expanding Prior Features
    sort_cols = ['reader_id', 'session_date']
    if 'start_time' in df.columns:
        sort_cols.append('start_time')
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)
    
    # Compute expanding prior aggregations per reader (shift by 1 so target session outcome is NOT included)
    grouped = df_sorted.groupby('reader_id')
    
    df_sorted['reader_prior_sessions_count'] = grouped.cumcount()
    df_sorted['reader_prior_completion_rate'] = grouped['completed'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.0)
    df_sorted['reader_prior_avg_duration'] = grouped['duration_minutes'].transform(lambda x: x.shift(1).expanding().mean()).fillna(df_sorted['duration_minutes'].median())
    df_sorted['reader_prior_avg_pages'] = grouped['pages_read'].transform(lambda x: x.shift(1).expanding().mean()).fillna(df_sorted['pages_read'].median())
    df_sorted['reader_prior_avg_ppm'] = grouped['pages_per_minute'].transform(lambda x: x.shift(1).expanding().mean()).fillna(df_sorted['pages_per_minute'].median())
    
    # Gap in days since last session
    df_sorted['prev_session_date'] = grouped['session_date'].shift(1)
    df_sorted['reader_days_since_last_session'] = (df_sorted['session_date'] - df_sorted['prev_session_date']).dt.days.fillna(0.0)
    df_sorted.drop(columns=['prev_session_date'], inplace=True, errors='ignore')
    
    # 3. Global Reader-level Aggregations (Strictly for Reader Profiling & Clustering)
    reader_agg = df.groupby('reader_id').agg(
        reader_total_sessions=('session_id', 'count'),
        reader_avg_session_duration=('duration_minutes', 'mean'),
        reader_avg_pages_per_session=('pages_read', 'mean'),
        reader_avg_speed_ppm=('pages_per_minute', 'mean'),
        reader_completion_rate=('completed', 'mean'),
        reader_unique_books=('book_id', 'nunique'),
        reader_first_date=('session_date', 'min'),
        reader_last_date=('session_date', 'max')
    ).reset_index()
    
    reader_agg['reader_abandonment_rate'] = 1.0 - reader_agg['reader_completion_rate']
    
    # Active days & reading frequency (sessions per week)
    reader_agg['active_days_span'] = (reader_agg['reader_last_date'] - reader_agg['reader_first_date']).dt.days.clip(lower=1)
    reader_agg['reading_frequency_per_week'] = (reader_agg['reader_total_sessions'] / reader_agg['active_days_span']) * 7.0
    
    reader_gaps = df_sorted.groupby('reader_id')['reader_days_since_last_session'].mean().reset_index()
    reader_gaps.rename(columns={'reader_days_since_last_session': 'average_gap_between_sessions'}, inplace=True)
    
    reader_features = reader_agg.merge(reader_gaps, on='reader_id', how='left')
    
    # Merge global summary features for backward compatibility with clustering
    df_engineered = df_sorted.merge(
        reader_features[['reader_id', 'reading_frequency_per_week', 'average_gap_between_sessions']], 
        on='reader_id', 
        how='left'
    )
    
    return df_engineered, reader_features

def build_ml_feature_set(df_engineered):
    """
    Transforms engineered dataset into a clean matrix ready for ML classification model training.
    Uses ONLY leak-free features (no target leakage or future dataset information).
    Target: completed (0/1)
    """
    df = df_engineered.copy()
    
    cat_cols = ['format', 'genre_clean', 'genre', 'age_group', 'time_of_day', 'reading_experience', 'preferred_format', 'format_availability', 'device_type', 'location_type']
    present_cat_cols = [c for c in cat_cols if c in df.columns]
    
    df_ml = pd.get_dummies(
        df, 
        columns=present_cat_cols,
        drop_first=True,
        dtype=int
    )
    
    # Strictly leak-free numeric features
    numeric_features = [
        'page_count', 'duration_minutes', 'pages_read', 'pages_per_minute',
        'reader_prior_avg_duration', 'reader_prior_avg_pages', 'reader_prior_avg_ppm',
        'reader_prior_completion_rate', 'reader_prior_sessions_count',
        'reader_days_since_last_session', 'reading_frequency_per_week',
        'average_gap_between_sessions', 'books_per_month'
    ]
    
    valid_numeric = [c for c in numeric_features if c in df_ml.columns]
    dummy_cols = [c for c in df_ml.columns if any(c.startswith(f"{prefix}_") for prefix in present_cat_cols)]
    
    feature_cols = valid_numeric + dummy_cols
    
    X = df_ml[feature_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df_ml['completed'].astype(int)
    
    return X, y, feature_cols

if __name__ == "__main__":
    from src.data_processing import process_and_clean_data
    _, _, _, df_master, _ = process_and_clean_data()
    df_eng, r_feat = calculate_behavioral_features(df_master)
    print(f"[Feature Engineering] Engine completed. Master features shape: {df_eng.shape}")
