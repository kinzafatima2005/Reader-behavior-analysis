import os
from src.data_generator import save_generated_data
from src.data_processing import process_and_clean_data
from src.sql_queries import build_database, get_completion_rate_by_genre
from src.feature_engineering import calculate_behavioral_features, build_ml_feature_set
from src.analysis import run_all_statistical_tests
from src.modeling import train_and_evaluate_models, save_trained_pipeline
from src.clustering import perform_reader_clustering, save_clustering_artifacts

def run_pipeline():
    print("==========================================================")
    print("      READING BEHAVIOR ANALYTICS PIPELINE RUNNER          ")
    print("==========================================================")
    
    # 1. Data Generation
    print("\n[Step 1/6] Synthesizing raw datasets...")
    save_generated_data()
    
    # 2. Data Processing & Cleaning
    print("\n[Step 2/6] Cleaning and preprocessing data...")
    df_readers, df_books, df_sessions, df_master, report = process_and_clean_data()
    
    # 3. SQLite Database Build
    print("\n[Step 3/6] Populating SQLite database (reading_data.db)...")
    build_database()
    
    # 4. Feature Engineering & Statistical Analysis
    print("\n[Step 4/6] Engineering behavioral metrics & running statistical tests...")
    df_eng, r_features = calculate_behavioral_features(df_master)
    stat_results = run_all_statistical_tests(df_master)
    print(f" - Speed Test T-Stat: {stat_results['speed_by_format']['t_statistic']}, p-val: {stat_results['speed_by_format']['p_value_ttest']:.4e}")
    print(f" - Completion Test Chi2: {stat_results['completion_by_format']['chi2_statistic']}, p-val: {stat_results['completion_by_format']['p_value']:.4e}")
    
    # 5. Machine Learning & Explainable AI
    print("\n[Step 5/6] Training book completion ML classifier models...")
    X, y, feature_cols = build_ml_feature_set(df_eng)
    eval_res = train_and_evaluate_models(X, y)
    print(f" - Best ML Model: {eval_res['best_model_name']}")
    print(eval_res['results_summary'])
    save_trained_pipeline(eval_res['best_model'], eval_res['scaler'], feature_cols)
    
    # 6. Unsupervised Reader Clustering
    print("\n[Step 6/6] Executing unsupervised reader segmentation...")
    c_res = perform_reader_clustering(r_features)
    print(f" - Silhouette Score (KMeans): {c_res['silhouette_kmeans']}")
    print(" - Derived Archetypes:", list(c_res['archetype_map'].values()))
    save_clustering_artifacts(c_res)
    
    print("\n==========================================================")
    print(" PIPELINE SUCCESSFULLY EXECUTED! ALL ASSETS READY.        ")
    print("==========================================================")

if __name__ == "__main__":
    run_pipeline()
