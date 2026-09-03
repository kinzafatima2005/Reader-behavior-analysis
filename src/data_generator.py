import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_reading_dataset(
    num_readers=350,
    num_books=150,
    num_sessions=3500,
    seed=42
):
    """
    Generates a multi-level reading behavior dataset with realistic relationships,
    demographics, session logs, survey data, and intentional real-world anomalies.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # ---------------------------------------------------------
    # 1. Readers Dataset (Survey & Profile Info)
    # ---------------------------------------------------------
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    age_probs = [0.20, 0.35, 0.25, 0.12, 0.08]
    experiences = ['Beginner', 'Intermediate', 'Advanced']
    formats = ['Physical', 'Ebook', 'No Preference']
    
    readers = []
    for r_idx in range(1, num_readers + 1):
        r_id = f"R{r_idx:04d}"
        age_grp = np.random.choice(age_groups, p=age_probs)
        exp = np.random.choice(experiences, p=[0.25, 0.50, 0.25])
        pref_fmt = np.random.choice(formats, p=[0.45, 0.45, 0.10])
        bpm = int(np.random.poisson(lam=3.5 if exp == 'Advanced' else (2.0 if exp == 'Intermediate' else 1.0)))
        bpm = max(1, bpm)
        
        # Survey responses
        survey_speed = round(float(np.random.normal(loc=0.6, scale=0.15)), 2) # pages per min
        owns_ereader = 1 if pref_fmt == 'Ebook' else int(np.random.choice([0, 1], p=[0.7, 0.3]))
        avg_session_pref_min = int(np.random.choice([15, 30, 45, 60, 90], p=[0.1, 0.3, 0.35, 0.2, 0.05]))
        
        readers.append({
            'reader_id': r_id,
            'age_group': age_grp,
            'reading_experience': exp,
            'books_per_month': bpm,
            'preferred_format': pref_fmt,
            'survey_speed_ppm': survey_speed,
            'owns_ereader': owns_ereader,
            'typical_session_min': avg_session_pref_min,
            'data_source': 'Survey_Collected'
        })
    
    df_readers = pd.DataFrame(readers)
    
    # ---------------------------------------------------------
    # 2. Books Dataset
    # ---------------------------------------------------------
    genres = ['Fiction', 'Non-Fiction', 'Mystery', 'Science Fiction', 'Fantasy', 'Romance', 'Biography', 'Self-Help', 'Academic']
    # Intentional inconsistent genre strings to simulate raw data
    raw_genres_pool = genres + ['Sci-Fi', 'SciFi', 'non_fiction', 'Self Help']
    
    authors = [
        "A. E. Sterling", "B. K. Thorne", "C. R. Vance", "D. M. Chen", 
        "E. J. Harrison", "F. L. Mercer", "G. H. Wells", "H. P. Craft",
        "I. M. Banks", "J. K. Rowling", "K. N. Robinson", "L. T. Torres"
    ]
    
    title_prefixes = ["The Silent", "Echoes of", "Principles of", "Shadows over", "Beyond the", "Chronicles of", "The Art of", "Guide to"]
    title_suffixes = ["Time", "Mind", "Eternity", "Data", "Wisdom", "Stars", "Empire", "Solitude", "Thinking"]
    
    books = []
    for b_idx in range(1, num_books + 1):
        b_id = f"B{b_idx:03d}"
        title = f"{random.choice(title_prefixes)} {random.choice(title_suffixes)}"
        author = random.choice(authors)
        # Raw genre (with some dirty strings for cleaning pipeline)
        genre = random.choice(raw_genres_pool)
        pub_year = int(np.random.randint(1980, 2026))
        page_count = int(np.random.randint(120, 750))
        fmt_availability = np.random.choice(['Physical', 'Ebook', 'Both'], p=[0.3, 0.3, 0.4])
        
        books.append({
            'book_id': b_id,
            'title': title,
            'author': author,
            'genre': genre,
            'publication_year': pub_year,
            'page_count': page_count,
            'format_availability': fmt_availability,
            'data_source': 'API_Catalog'
        })
        
    df_books = pd.DataFrame(books)
    
    # ---------------------------------------------------------
    # 3. Sessions Dataset (Synthetic Reading Logs)
    # ---------------------------------------------------------
    start_date = datetime(2026, 1, 1)
    sessions = []
    s_idx = 1
    
    # Assign reading assignments to readers
    for _, reader in df_readers.iterrows():
        r_id = reader['reader_id']
        # Number of books attempted by this reader
        num_reader_books = int(np.random.randint(3, 10))
        sampled_books = df_books.sample(n=num_reader_books, replace=True)
        
        reader_start_day = int(np.random.randint(0, 30))
        current_date = start_date + timedelta(days=reader_start_day)
        
        for _, book in sampled_books.iterrows():
            b_id = book['book_id']
            page_count = book['page_count']
            
            # Reader's tendency to finish this book (based on experience and format preference)
            will_eventually_finish = np.random.choice([1, 0], p=[0.65, 0.35])
            
            # Format selection
            if book['format_availability'] == 'Physical':
                session_format = 'Physical'
            elif book['format_availability'] == 'Ebook':
                session_format = 'Ebook'
            else:
                session_format = reader['preferred_format'] if reader['preferred_format'] != 'No Preference' else random.choice(['Physical', 'Ebook'])
                
            device_type = 'Physical Book' if session_format == 'Physical' else np.random.choice(['Kindle', 'Tablet', 'Phone'], p=[0.55, 0.30, 0.15])
            
            total_pages_read = 0
            # Generate 2 to 10 sessions for this book
            max_sessions = int(np.random.randint(2, 12))
            
            for sess_num in range(max_sessions):
                s_id = f"S{s_idx:05d}"
                s_idx += 1
                
                # Session time advance (gap between sessions)
                gap_days = int(np.random.choice([0, 1, 2, 3, 5, 7], p=[0.3, 0.3, 0.2, 0.1, 0.06, 0.04]))
                current_date += timedelta(days=gap_days)
                
                hour_probs = np.array([
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.02, # 0-5
                    0.04, 0.06, 0.06, 0.05, 0.04, 0.04, # 6-11
                    0.05, 0.05, 0.04, 0.04, 0.05, 0.06, # 12-17
                    0.08, 0.09, 0.09, 0.06, 0.02, 0.01  # 18-23
                ])
                hour_probs /= hour_probs.sum()
                hour = int(np.random.choice(range(24), p=hour_probs))
                minute = int(np.random.randint(0, 60))
                start_time_str = f"{hour:02d}:{minute:02d}"
                
                base_duration = np.random.normal(loc=38 if session_format == 'Physical' else 32, scale=10)
                duration_min = max(8.0, round(float(base_duration), 1))
                
                speed_mult = 1.12 if session_format == 'Ebook' else 1.0
                speed = max(0.2, reader['survey_speed_ppm'] * speed_mult)
                pages_read = int(max(2, round(duration_min * np.random.normal(loc=speed, scale=0.08))))
                
                total_pages_read += pages_read
                
                if total_pages_read >= page_count or (will_eventually_finish and sess_num == max_sessions - 1):
                    total_pages_read = page_count
                    pct_completed = 100.0
                    completed = 1
                else:
                    pct_completed = round((total_pages_read / page_count) * 100.0, 2)
                    completed = 1 if pct_completed >= 95.0 else 0
                    
                location_type = np.random.choice(['Home', 'Commute', 'Library', 'Outdoor', 'Coffee Shop'], p=[0.50, 0.20, 0.15, 0.08, 0.07])
                
                sessions.append({
                    'session_id': s_id,
                    'reader_id': r_id,
                    'book_id': b_id,
                    'session_date': current_date.strftime('%Y-%m-%d'),
                    'start_time': start_time_str,
                    'duration_minutes': duration_min,
                    'pages_read': pages_read,
                    'percentage_completed': pct_completed,
                    'format': session_format,
                    'device_type': device_type,
                    'location_type': location_type,
                    'completed': completed,
                    'data_source': 'Synthetic_Session_Log'
                })
                
                if completed == 1:
                    break # Finished book, move to next book
                    
    df_sessions = pd.DataFrame(sessions)
    
    # ---------------------------------------------------------
    # 4. Inject Realistic Noise / Anomalies (for cleaning validation)
    # ---------------------------------------------------------
    # Missing values in location_type (5% missing)
    missing_loc_idx = df_sessions.sample(frac=0.05, random_state=seed).index
    df_sessions.loc[missing_loc_idx, 'location_type'] = np.nan
    
    # Negative / Zero duration anomaly (0.5% records)
    bad_dur_idx = df_sessions.sample(frac=0.005, random_state=seed).index
    df_sessions.loc[bad_dur_idx, 'duration_minutes'] = -10.0
    
    # Extreme outlier duration (e.g. forgot to close app - 500 mins) (0.3% records)
    outlier_dur_idx = df_sessions.sample(frac=0.003, random_state=seed+1).index
    df_sessions.loc[outlier_dur_idx, 'duration_minutes'] = 520.0
    
    # Duplicate session logs (0.8% records)
    dupes = df_sessions.sample(frac=0.008, random_state=seed+2)
    df_sessions = pd.concat([df_sessions, dupes], ignore_index=True)
    
    return df_readers, df_books, df_sessions

def save_generated_data(output_dir="data/raw"):
    """
    Saves readers, books, and sessions datasets into CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_readers, df_books, df_sessions = generate_reading_dataset()
    
    readers_path = os.path.join(output_dir, "readers_raw.csv")
    books_path = os.path.join(output_dir, "books_raw.csv")
    sessions_path = os.path.join(output_dir, "sessions_raw.csv")
    
    df_readers.to_csv(readers_path, index=False)
    df_books.to_csv(books_path, index=False)
    df_sessions.to_csv(sessions_path, index=False)
    
    print(f"[Data Generator] Datasets saved successfully to '{output_dir}':")
    print(f" - Readers: {len(df_readers)} records")
    print(f" - Books: {len(df_books)} records")
    print(f" - Sessions: {len(df_sessions)} records")

if __name__ == "__main__":
    save_generated_data()
