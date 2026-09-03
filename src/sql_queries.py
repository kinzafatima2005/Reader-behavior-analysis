import os
import sqlite3
import pandas as pd

def build_database(
    processed_dir="data/processed",
    db_path="database/reading_data.db",
    schema_path="database/schema.sql"
):
    """
    Initializes SQLite database and populates tables from processed CSV datasets.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Execute DDL
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            cursor.executescript(f.read())
            
    # Load processed CSVs
    df_readers = pd.read_csv(os.path.join(processed_dir, "readers_clean.csv"))
    df_books = pd.read_csv(os.path.join(processed_dir, "books_clean.csv"))
    df_sessions = pd.read_csv(os.path.join(processed_dir, "sessions_clean.csv"))
    
    # Standardize column mapping for books
    if 'genre_clean' in df_books.columns:
        df_books['genre'] = df_books['genre_clean']
        df_books = df_books.drop(columns=['genre_clean'])
        
    df_readers.to_sql('readers', conn, if_exists='replace', index=False)
    df_books.to_sql('books', conn, if_exists='replace', index=False)
    df_sessions.to_sql('sessions', conn, if_exists='replace', index=False)
    
    conn.commit()
    conn.close()
    print(f"[SQL Database] Database successfully created & populated at '{db_path}'.")

def execute_query(query_sql, db_path="database/reading_data.db"):
    """
    Executes a SQL query against reading_data.db and returns a pandas DataFrame.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query_sql, conn)
    finally:
        conn.close()
    return df

# Analytical SQL Queries
def get_completion_rate_by_genre(db_path="database/reading_data.db"):
    sql = """
    SELECT 
        b.genre,
        COUNT(DISTINCT s.book_id) AS total_books,
        COUNT(s.session_id) AS total_sessions,
        ROUND(AVG(s.completed) * 100.0, 2) AS completion_rate_pct,
        ROUND(AVG(s.pages_per_minute), 2) AS avg_reading_speed_ppm
    FROM sessions s
    JOIN books b ON s.book_id = b.book_id
    GROUP BY b.genre
    ORDER BY completion_rate_pct DESC;
    """
    return execute_query(sql, db_path)

def get_format_pages_per_session(db_path="database/reading_data.db"):
    sql = """
    SELECT 
        format,
        COUNT(session_id) AS total_sessions,
        ROUND(AVG(pages_read), 2) AS avg_pages_per_session,
        ROUND(AVG(duration_minutes), 2) AS avg_duration_minutes,
        ROUND(AVG(pages_per_minute), 2) AS avg_speed_ppm,
        ROUND(AVG(completed) * 100.0, 2) AS completion_rate_pct
    FROM sessions
    GROUP BY format;
    """
    return execute_query(sql, db_path)

def get_top_readers(db_path="database/reading_data.db", limit=10):
    sql = f"""
    SELECT 
        r.reader_id,
        r.age_group,
        r.reading_experience,
        r.preferred_format,
        COUNT(s.session_id) AS total_sessions,
        SUM(s.pages_read) AS total_pages_read,
        ROUND(AVG(s.completed) * 100.0, 2) AS completion_rate_pct
    FROM readers r
    JOIN sessions s ON r.reader_id = s.reader_id
    GROUP BY r.reader_id
    HAVING total_sessions >= 5
    ORDER BY completion_rate_pct DESC, total_pages_read DESC
    LIMIT {limit};
    """
    return execute_query(sql, db_path)

def get_speed_by_genre(db_path="database/reading_data.db"):
    sql = """
    SELECT 
        b.genre,
        ROUND(AVG(s.pages_per_minute), 3) AS avg_pages_per_minute,
        ROUND(MIN(s.pages_per_minute), 3) AS min_speed,
        ROUND(MAX(s.pages_per_minute), 3) AS max_speed,
        ROUND(AVG(s.duration_minutes), 1) AS avg_session_duration
    FROM sessions s
    JOIN books b ON s.book_id = b.book_id
    GROUP BY b.genre
    ORDER BY avg_pages_per_minute DESC;
    """
    return execute_query(sql, db_path)

def get_monthly_activity(db_path="database/reading_data.db"):
    sql = """
    SELECT 
        STRFTIME('%Y-%m', session_date) AS month,
        COUNT(session_id) AS total_sessions,
        SUM(pages_read) AS total_pages_read,
        ROUND(AVG(duration_minutes), 2) AS avg_duration_min,
        ROUND(AVG(completed) * 100.0, 2) AS completion_rate_pct
    FROM sessions
    GROUP BY month
    ORDER BY month ASC;
    """
    return execute_query(sql, db_path)

def get_books_most_sessions(db_path="database/reading_data.db", limit=10):
    sql = f"""
    SELECT 
        b.book_id,
        b.title,
        b.author,
        b.genre,
        b.page_count,
        COUNT(s.session_id) AS total_sessions,
        SUM(s.pages_read) AS pages_read_sum,
        ROUND(AVG(s.completed) * 100.0, 1) AS completion_rate_pct
    FROM books b
    JOIN sessions s ON b.book_id = s.book_id
    GROUP BY b.book_id
    ORDER BY total_sessions DESC
    LIMIT {limit};
    """
    return execute_query(sql, db_path)

def get_abandonment_percentage(db_path="database/reading_data.db"):
    sql = """
    SELECT 
        format,
        COUNT(session_id) AS total_sessions,
        SUM(CASE WHEN completed = 0 THEN 1 ELSE 0 END) AS abandoned_sessions,
        ROUND((SUM(CASE WHEN completed = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(session_id)), 2) AS abandonment_rate_pct
    FROM sessions
    GROUP BY format;
    """
    return execute_query(sql, db_path)

if __name__ == "__main__":
    build_database()
    print("Testing completion rate by genre query:")
    print(get_completion_rate_by_genre().head())
