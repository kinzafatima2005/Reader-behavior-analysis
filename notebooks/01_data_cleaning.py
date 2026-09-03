"""
Notebook 01: Data Cleaning & Preprocessing
Applies data cleaning logic, outlier flagging, missing value handling, and schema validation.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generator import save_generated_data
from src.data_processing import process_and_clean_data

print("Generating raw data...")
save_generated_data()

print("\nRunning cleaning pipeline...")
df_readers, df_books, df_sessions, df_master, report = process_and_clean_data()

print("\n--- Cleaning Report Summary ---")
for metric, count in report.items():
    print(f"  {metric}: {count}")

print("\nMaster Dataset Shape:", df_master.shape)
print("\nSample Master Data:")
print(df_master[['session_id', 'reader_id', 'format', 'pages_read', 'duration_minutes', 'pages_per_minute', 'completed']].head())
