-- Schema definition for Reading Behavior Analytics Database

DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS readers;

CREATE TABLE readers (
    reader_id VARCHAR(20) PRIMARY KEY,
    age_group VARCHAR(20),
    reading_experience VARCHAR(20),
    books_per_month INT,
    preferred_format VARCHAR(20),
    survey_speed_ppm FLOAT,
    owns_ereader INT,
    typical_session_min INT,
    data_source VARCHAR(30)
);

CREATE TABLE books (
    book_id VARCHAR(20) PRIMARY KEY,
    title VARCHAR(200),
    author VARCHAR(100),
    genre VARCHAR(50),
    publication_year INT,
    page_count INT,
    format_availability VARCHAR(20),
    data_source VARCHAR(30)
);

CREATE TABLE sessions (
    session_id VARCHAR(20) PRIMARY KEY,
    reader_id VARCHAR(20),
    book_id VARCHAR(20),
    session_date DATE,
    start_time VARCHAR(10),
    duration_minutes FLOAT,
    pages_read INT,
    percentage_completed FLOAT,
    format VARCHAR(20),
    device_type VARCHAR(30),
    location_type VARCHAR(50),
    completed INT,
    pages_per_minute FLOAT,
    time_of_day VARCHAR(20),
    day_of_week VARCHAR(20),
    data_source VARCHAR(30),
    FOREIGN KEY (reader_id) REFERENCES readers(reader_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);
