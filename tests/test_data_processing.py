import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.data_generator import generate_reading_dataset
from src.data_processing import clean_genre, process_and_clean_data

class TestDataProcessing(unittest.TestCase):
    def test_generate_reading_dataset(self):
        df_readers, df_books, df_sessions = generate_reading_dataset(num_readers=20, num_books=10)
        self.assertEqual(len(df_readers), 20)
        self.assertEqual(len(df_books), 10)
        self.assertTrue(len(df_sessions) >= 20)
        self.assertIn('reader_id', df_readers.columns)
        self.assertIn('book_id', df_books.columns)
        self.assertIn('session_id', df_sessions.columns)

    def test_clean_genre(self):
        self.assertEqual(clean_genre('sci-fi'), 'Science Fiction')
        self.assertEqual(clean_genre('SciFi'), 'Science Fiction')
        self.assertEqual(clean_genre('non_fiction'), 'Non-Fiction')
        self.assertEqual(clean_genre('self help'), 'Self-Help')
        self.assertEqual(clean_genre('Fiction'), 'Fiction')

    def test_no_missing_values_after_preprocessing(self):
        df_readers, df_books, df_sessions, df_master, report = process_and_clean_data()
        self.assertEqual(df_sessions['location_type'].isna().sum(), 0)
        self.assertTrue(all(df_sessions['duration_minutes'] > 0))
        self.assertFalse(df_master['genre_clean'].isna().any())

    def test_valid_completion_values(self):
        _, _, df_sessions, _, _ = process_and_clean_data()
        unique_vals = set(df_sessions['completed'].unique())
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_valid_formats(self):
        _, _, df_sessions, _, _ = process_and_clean_data()
        unique_fmts = set(df_sessions['format'].unique())
        self.assertTrue(unique_fmts.issubset({'Physical', 'Ebook'}))

    def test_pages_per_minute_calculation(self):
        _, _, df_sessions, _, _ = process_and_clean_data()
        calculated_ppm = df_sessions['pages_read'] / df_sessions['duration_minutes']
        np.testing.assert_allclose(df_sessions['pages_per_minute'].values, calculated_ppm.values, rtol=1e-5)

    def test_outlier_flagging(self):
        _, _, df_sessions, _, _ = process_and_clean_data()
        self.assertIn('is_duration_outlier', df_sessions.columns)
        self.assertIn('is_speed_outlier', df_sessions.columns)

    def test_no_duplicate_session_ids(self):
        _, _, df_sessions, _, _ = process_and_clean_data()
        dupes = df_sessions.duplicated(subset=['session_id']).sum()
        self.assertEqual(dupes, 0)

if __name__ == "__main__":
    unittest.main()
