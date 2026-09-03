"""
Notebook 04: Machine Learning & Explainable AI
Trains predictive classifiers for book completion and evaluates performance & SHAP feature importances.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import process_and_clean_data
from src.feature_engineering import calculate_behavioral_features, build_ml_feature_set
from src.modeling import train_and_evaluate_models, get_feature_importances, explain_single_prediction

_, _, _, df_master, _ = process_and_clean_data()
df_eng, _ = calculate_behavioral_features(df_master)
X, y, feature_cols = build_ml_feature_set(df_eng)

print("Training classification models...")
eval_res = train_and_evaluate_models(X, y)

print("\n--- Classifier Evaluation Summary ---")
print(eval_res['results_summary'])

print(f"\nBest Model: {eval_res['best_model_name']}")

df_imp = get_feature_importances(eval_res['best_model'], feature_cols)
print("\n--- Top 10 Feature Importances ---")
print(df_imp.head(10).to_string(index=False))

# Explain single prediction example
sample_instance = X.iloc[[0]]
explanation = explain_single_prediction(eval_res['best_model'], eval_res['scaler'], sample_instance, feature_cols)
print("\n--- Explainable AI Single Instance Prediction ---")
print(f"Predicted Completion Probability: {explanation['completion_pct']}%")
print(f"Factors Increasing Completion: {explanation['increasing_factors']}")
print(f"Factors Decreasing Completion: {explanation['decreasing_factors']}")
