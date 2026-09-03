import os
import pandas as pd
import numpy as np

def clean_genre(genre_str):
    """
    Standardizes inconsistent genre tags.
    """
    if not isinstance(genre_str, str):
        return 'Other'
    g_lower = genre_str.strip().lower()
    if g_lower in ['sci-fi', 'scifi', 'science fiction']:
        return 'Science Fiction'
    elif g_lower in ['non_fiction', 'non fiction', 'non-fiction']:
        return 'Non-Fiction'
    elif g_lower in ['self help', 'self-help']:
        return 'Self-Help'
    elif 'fiction' in g_lower and g_lower != 'science fiction':
        return 'Fiction'
    else:
        return genre_str.strip().title()

def process_and_clean_data(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Cleans raw reading datasets, handles missing values, removes duplicates,
    standardizes categories, flags outliers, and outputs processed datasets.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    readers_path = os.path.join(raw_dir, "readers_raw.csv")
    books_path = os.path.join(raw_dir, "books_raw.csv")
    sessions_path = os.path.join(raw_dir, "sessions_raw.csv")
    
    df_readers = pd.read_csv(readers_path)
    df_books = pd.read_csv(books_path)
    df_sessions = pd.read_csv(sessions_path)
    
    cleaning_report = {
        'initial_sessions_count': len(df_sessions),
        'duplicates_removed': 0,
        'missing_locations_imputed': 0,
        'invalid_durations_corrected': 0,
        'duration_outliers_flagged': 0,
        'impossible_speeds_flagged': 0
    }
    
    # 1. Clean Books dataset
    df_books['genre_clean'] = df_books['genre'].apply(clean_genre)
    
    # 2. Clean Sessions dataset - Remove duplicates
    session_cols = ['session_id', 'reader_id', 'book_id', 'session_date', 'start_time']
    dupes_count = df_sessions.duplicated(subset=session_cols).sum()
    df_sessions = df_sessions.drop_duplicates(subset=session_cols).copy()
    cleaning_report['duplicates_removed'] = int(dupes_count)
    
    # 3. Handle missing location types
    missing_locs = df_sessions['location_type'].isna().sum()
    df_sessions['location_type'] = df_sessions['location_type'].fillna('Unknown / Unspecified')
    cleaning_report['missing_locations_imputed'] = int(missing_locs)
    
    # 4. Handle invalid duration values (<= 0 minutes)
    invalid_dur_mask = df_sessions['duration_minutes'] <= 0
    cleaning_report['invalid_durations_corrected'] = int(invalid_dur_mask.sum())
    median_duration = df_sessions.loc[~invalid_dur_mask, 'duration_minutes'].median()
    df_sessions.loc[invalid_dur_mask, 'duration_minutes'] = median_duration
    
    # 5. Calculate pages per minute and handle extreme reading speed outliers
    df_sessions['pages_per_minute'] = df_sessions['pages_read'] / df_sessions['duration_minutes']
    
    # Flag extreme duration outliers (> 300 minutes, likely unclosed apps)
    df_sessions['is_duration_outlier'] = (df_sessions['duration_minutes'] > 300).astype(int)
    cleaning_report['duration_outliers_flagged'] = int(df_sessions['is_duration_outlier'].sum())
    
    # Flag impossible reading speeds (> 3.5 pages/min)
    df_sessions['is_speed_outlier'] = (df_sessions['pages_per_minute'] > 3.5).astype(int)
    cleaning_report['impossible_speeds_flagged'] = int(df_sessions['is_speed_outlier'].sum())
    
    # Time of Day Classification
    def get_time_of_day(time_str):
        try:
            hour = int(str(time_str).split(':')[0])
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        except Exception:
            return 'Evening'
            
    df_sessions['time_of_day'] = df_sessions['start_time'].apply(get_time_of_day)
    df_sessions['session_date'] = pd.to_datetime(df_sessions['session_date'])
    df_sessions['day_of_week'] = df_sessions['session_date'].dt.day_name()
    
    # Save Clean Datasets
    readers_clean_path = os.path.join(processed_dir, "readers_clean.csv")
    books_clean_path = os.path.join(processed_dir, "books_clean.csv")
    sessions_clean_path = os.path.join(processed_dir, "sessions_clean.csv")
    
    df_readers.to_csv(readers_clean_path, index=False)
    df_books.to_csv(books_clean_path, index=False)
    df_sessions.to_csv(sessions_clean_path, index=False)
    
    # Create Merged Master Analytical Dataset
    df_master = df_sessions.merge(df_readers, on='reader_id', how='left')
    df_master = df_master.merge(df_books, on='book_id', how='left')
    
    master_path = os.path.join(processed_dir, "reading_sessions_master.csv")
    df_master.to_csv(master_path, index=False)
    
    print(f"[Data Processing] Data cleaning complete. Cleaning summary:")
    for k, v in cleaning_report.items():
        print(f" - {k}: {v}")
    print(f"Master merged dataset saved to '{master_path}' ({len(df_master)} rows)")
    
    return df_readers, df_books, df_sessions, df_master, cleaning_report

if __name__ == "__main__":
    process_and_clean_data()
