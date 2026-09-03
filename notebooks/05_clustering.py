"""
Notebook 05: Unsupervised Clustering & Reader Segmentation
Applies K-Means and Hierarchical Clustering to profile reader archetypes.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import process_and_clean_data
from src.feature_engineering import calculate_behavioral_features
from src.clustering import perform_reader_clustering

_, _, _, df_master, _ = process_and_clean_data()
_, r_feat = calculate_behavioral_features(df_master)

c_res = perform_reader_clustering(r_feat)

print("--- Reader Clustering Evaluation ---")
print(f"K-Means Silhouette Score: {c_res['silhouette_kmeans']}")
print(f"Hierarchical Silhouette Score: {c_res['silhouette_agg']}")

print("\n--- Derived Reader Archetypes Map ---")
for cid, label in c_res['archetype_map'].items():
    print(f"  {label}")

print("\n--- Cluster Profile Centroids ---")
print(c_res['cluster_profiles'])
