import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.feature_engineering import calculate_behavioral_features, build_ml_feature_set

class TestFeatureEngineering(unittest.TestCase):
    def test_calculate_behavioral_features(self):
        df_sample = pd.DataFrame([{
            'session_id': 'S001',
            'reader_id': 'R001',
            'book_id': 'B001',
            'session_date': '2026-01-01',
            'start_time': '10:00',
            'duration_minutes': 40.0,
            'pages_read': 20,
            'completed': 1,
            'format': 'Ebook',
            'genre': 'Fiction',
            'age_group': '25-34',
            'time_of_day': 'Morning',
            'reading_experience': 'Intermediate',
            'books_per_month': 3
        }])
        
        df_eng, r_feat = calculate_behavioral_features(df_sample)
        self.assertIn('pages_per_minute', df_eng.columns)
        self.assertAlmostEqual(df_eng['pages_per_minute'].iloc[0], 0.5)
        self.assertEqual(len(r_feat), 1)

    def test_leak_free_prior_features(self):
        df_two_sessions = pd.DataFrame([
            {
                'session_id': 'S001',
                'reader_id': 'R001',
                'book_id': 'B001',
                'session_date': '2026-01-01',
                'start_time': '10:00',
                'duration_minutes': 30.0,
                'pages_read': 15,
                'completed': 1,
                'format': 'Ebook',
                'genre': 'Fiction'
            },
            {
                'session_id': 'S002',
                'reader_id': 'R001',
                'book_id': 'B001',
                'session_date': '2026-01-02',
                'start_time': '10:00',
                'duration_minutes': 40.0,
                'pages_read': 20,
                'completed': 0,
                'format': 'Ebook',
                'genre': 'Fiction'
            }
        ])
        
        df_eng, _ = calculate_behavioral_features(df_two_sessions)
        
        # First session prior completion rate must be 0.0 (no target leakage from session 1)
        self.assertEqual(df_eng['reader_prior_completion_rate'].iloc[0], 0.0)
        self.assertEqual(df_eng['reader_prior_sessions_count'].iloc[0], 0)
        
        # Second session prior completion rate must be 1.0 (from session 1 only)
        self.assertEqual(df_eng['reader_prior_completion_rate'].iloc[1], 1.0)
        self.assertEqual(df_eng['reader_prior_sessions_count'].iloc[1], 1)

    def test_build_ml_feature_set(self):
        df_sample = pd.DataFrame([{
            'session_id': 'S001',
            'reader_id': 'R001',
            'book_id': 'B001',
            'session_date': '2026-01-01',
            'start_time': '10:00',
            'duration_minutes': 40.0,
            'pages_read': 20,
            'page_count': 300,
            'completed': 1,
            'format': 'Ebook',
            'genre': 'Fiction',
            'age_group': '25-34',
            'time_of_day': 'Morning',
            'reading_experience': 'Intermediate',
            'books_per_month': 3,
            'reader_prior_avg_duration': 40.0,
            'reader_prior_avg_pages': 20.0,
            'reader_prior_avg_ppm': 0.5,
            'reader_prior_completion_rate': 0.0,
            'reader_prior_sessions_count': 0,
            'reader_days_since_last_session': 0.0,
            'reading_frequency_per_week': 3.5,
            'average_gap_between_sessions': 2.0
        }])
        
        X, y, feature_cols = build_ml_feature_set(df_sample)
        self.assertEqual(len(X), 1)
        self.assertEqual(len(y), 1)
        self.assertEqual(y.iloc[0], 1)
        self.assertNotIn('reader_completion_rate', feature_cols)

    def test_ml_feature_column_consistency(self):
        df_sample = pd.DataFrame([{
            'session_id': 'S001',
            'reader_id': 'R001',
            'book_id': 'B001',
            'session_date': '2026-01-01',
            'duration_minutes': 30.0,
            'pages_read': 15,
            'page_count': 250,
            'completed': 0,
            'format': 'Physical',
            'genre': 'Non-Fiction',
            'reader_prior_avg_duration': 30.0,
            'reader_prior_completion_rate': 0.0
        }])
        X, y, cols = build_ml_feature_set(df_sample)
        self.assertEqual(X.shape[1], len(cols))
        self.assertFalse(X.isna().any().any())

if __name__ == "__main__":
    unittest.main()
