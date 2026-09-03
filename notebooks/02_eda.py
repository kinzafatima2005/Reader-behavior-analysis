"""
Notebook 02: Exploratory Data Analysis
Investigates format differences, genre patterns, reader experience, and time of day trends.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data_processing import process_and_clean_data
from src.analysis import analyze_format_comparison, analyze_genre_comparison, analyze_time_of_day

_, _, _, df_master, _ = process_and_clean_data()

print("==================================================")
print(" 1. FORMAT COMPARISON (E-Book vs Physical)       ")
print("==================================================")
df_format = analyze_format_comparison(df_master)
print(df_format.to_string(index=False))

print("\n==================================================")
print(" 2. GENRE COMPARISON                             ")
print("==================================================")
df_genre = analyze_genre_comparison(df_master)
print(df_genre.to_string(index=False))

print("\n==================================================")
print(" 3. TIME OF DAY TRENDS                           ")
print("==================================================")
df_tod = analyze_time_of_day(df_master)
print(df_tod.to_string(index=False))
