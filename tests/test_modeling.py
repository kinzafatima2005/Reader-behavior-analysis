import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.modeling import train_and_evaluate_models, explain_single_prediction, get_feature_importances
from src.clustering import perform_reader_clustering
from src.recommendation import generate_behavioral_strategy

class TestModeling(unittest.TestCase):
    def test_ml_train_and_evaluate(self):
        np.random.seed(42)
        N = 100
        X = pd.DataFrame({
            'feature_1': np.random.randn(N),
            'feature_2': np.random.randn(N),
            'duration_minutes': np.random.uniform(10, 60, N)
        })
        y = pd.Series(np.random.choice([0, 1], size=N, p=[0.4, 0.6]))
        
        eval_res = train_and_evaluate_models(X, y)
        self.assertIn('results_summary', eval_res)
        self.assertIsNotNone(eval_res['best_model_name'])
        self.assertIn('accuracy', eval_res['results_summary'].columns)
        self.assertIn('f1_score', eval_res['results_summary'].columns)

    def test_baseline_model_evaluation(self):
        np.random.seed(42)
        N = 100
        X = pd.DataFrame({
            'feature_1': np.random.randn(N),
            'feature_2': np.random.randn(N)
        })
        y = pd.Series(np.random.choice([0, 1], size=N, p=[0.8, 0.2]))
        
        eval_res = train_and_evaluate_models(X, y)
        summary = eval_res['results_summary']
        self.assertIn('Majority Class Baseline', summary.index)
        self.assertTrue(summary.loc['Majority Class Baseline', 'accuracy'] > 0.5)

    def test_prediction_probability_bounds(self):
        np.random.seed(42)
        N = 50
        X = pd.DataFrame({
            'feature_1': np.random.randn(N),
            'feature_2': np.random.randn(N)
        })
        y = pd.Series(np.random.choice([0, 1], size=N))
        
        eval_res = train_and_evaluate_models(X, y)
        sample = X.iloc[[0]]
        explanation = explain_single_prediction(
            eval_res['best_model'], eval_res['scaler'], sample, list(X.columns)
        )
        
        self.assertTrue(0.0 <= explanation['completion_probability'] <= 1.0)
        self.assertTrue(0.0 <= explanation['completion_pct'] <= 100.0)

    def test_explain_single_prediction(self):
        np.random.seed(42)
        N = 50
        X = pd.DataFrame({
            'feature_1': np.random.randn(N),
            'feature_2': np.random.randn(N)
        })
        y = pd.Series(np.random.choice([0, 1], size=N))
        
        eval_res = train_and_evaluate_models(X, y)
        sample = X.iloc[[0]]
        explanation = explain_single_prediction(
            eval_res['best_model'], eval_res['scaler'], sample, list(X.columns)
        )
        
        self.assertIn('increasing_factors', explanation)
        self.assertIn('decreasing_factors', explanation)

    def test_feature_importance_extraction(self):
        np.random.seed(42)
        N = 50
        X = pd.DataFrame({
            'f1': np.random.randn(N),
            'f2': np.random.randn(N)
        })
        y = pd.Series(np.random.choice([0, 1], size=N))
        eval_res = train_and_evaluate_models(X, y)
        df_imp = get_feature_importances(eval_res['best_model'], list(X.columns))
        self.assertEqual(len(df_imp), 2)
        self.assertIn('importance', df_imp.columns)

    def test_clustering_archetype_mapping(self):
        df_r = pd.DataFrame({
            'reader_id': [f"R{i:03d}" for i in range(20)],
            'reading_frequency_per_week': np.random.uniform(1, 7, 20),
            'reader_avg_session_duration': np.random.uniform(15, 60, 20),
            'reader_avg_pages_per_session': np.random.uniform(10, 50, 20),
            'books_per_month': np.random.randint(1, 10, 20),
            'reader_completion_rate': np.random.uniform(0.2, 0.9, 20),
            'reader_avg_speed_ppm': np.random.uniform(0.3, 1.2, 20),
            'average_gap_between_sessions': np.random.uniform(1, 5, 20)
        })
        
        c_res = perform_reader_clustering(df_r, n_clusters=4)
        self.assertIn('silhouette_kmeans', c_res)
        self.assertEqual(len(c_res['archetype_map']), 4)

    def test_recommendation_strategy_generation(self):
        sample_reader = {
            'reader_avg_session_duration': 50.0,
            'reading_frequency_per_week': 2.1,
            'average_gap_between_sessions': 4.5,
            'preferred_format': 'Ebook',
            'reader_avg_speed_ppm': 0.65
        }
        strat = generate_behavioral_strategy(sample_reader)
        self.assertIn('recommended_session_window', strat)
        self.assertIn('behavioral_insights', strat)
        self.assertTrue(len(strat['behavioral_insights']) >= 3)

if __name__ == "__main__":
    unittest.main()
