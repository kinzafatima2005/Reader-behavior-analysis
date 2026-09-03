import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def perform_reader_clustering(reader_features_df, n_clusters=4, random_state=42):
    """
    Performs unsupervised clustering (K-Means & Hierarchical) on reader behavioral profiles.
    Identifies 4 behavioral clusters, which are subsequently interpreted as Daily Reader, 
    Weekend Reader, Speed Reader, and Abandoner based on centroid characteristics.
    """
    df_r = reader_features_df.copy()
    
    cluster_features = [
        'reading_frequency_per_week',
        'reader_avg_session_duration',
        'reader_avg_pages_per_session',
        'books_per_month',
        'reader_completion_rate',
        'reader_avg_speed_ppm',
        'average_gap_between_sessions'
    ]
    
    # Filter available features
    valid_cols = [c for c in cluster_features if c in df_r.columns]
    X_raw = df_r[valid_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    sil_kmeans = silhouette_score(X_scaled, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else 0.0
    
    # 2. Hierarchical Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X_scaled)
    sil_agg = silhouette_score(X_scaled, agg_labels) if len(np.unique(agg_labels)) > 1 else 0.0
    
    # 3. PCA 2D and 3D Projection
    pca_2d = PCA(n_components=2, random_state=random_state)
    pca_coords_2d = pca_2d.fit_transform(X_scaled)
    
    pca_3d = PCA(n_components=3, random_state=random_state)
    pca_coords_3d = pca_3d.fit_transform(X_scaled)
    
    df_r['cluster_kmeans'] = kmeans_labels
    df_r['cluster_hierarchical'] = agg_labels
    df_r['pca_x'] = pca_coords_2d[:, 0]
    df_r['pca_y'] = pca_coords_2d[:, 1]
    df_r['pca_z'] = pca_coords_3d[:, 2]
    
    # Derive Cluster Archetype Labels based on Centroids Ranking
    cluster_profiles = df_r.groupby('cluster_kmeans')[valid_cols].mean()
    
    # Sort cluster IDs by key dimensions to assign distinct archetypes
    freq_rank = cluster_profiles['reading_frequency_per_week'].sort_values(ascending=False).index.tolist()
    speed_rank = cluster_profiles['reader_avg_speed_ppm'].sort_values(ascending=False).index.tolist()
    dur_rank = cluster_profiles['reader_avg_session_duration'].sort_values(ascending=False).index.tolist()
    comp_rank = cluster_profiles['reader_completion_rate'].sort_values(ascending=True).index.tolist()
    
    archetype_map = {}
    assigned_labels = set()
    
    # 1. Highest frequency -> Daily Reader
    daily_c = freq_rank[0]
    archetype_map[daily_c] = f"Cluster {daily_c}: Daily Reader Archetype"
    assigned_labels.add(daily_c)
    
    # 2. Highest speed -> Speed Reader (if unassigned)
    speed_c = [c for c in speed_rank if c not in assigned_labels][0]
    archetype_map[speed_c] = f"Cluster {speed_c}: Speed Reader Archetype"
    assigned_labels.add(speed_c)
    
    # 3. Lowest completion / highest gap -> Abandoner (if unassigned)
    abandon_c = [c for c in comp_rank if c not in assigned_labels][0]
    archetype_map[abandon_c] = f"Cluster {abandon_c}: Abandoner Archetype"
    assigned_labels.add(abandon_c)
    
    # 4. Remaining -> Weekend Reader
    remaining_c = [c for c in cluster_profiles.index if c not in assigned_labels][0]
    archetype_map[remaining_c] = f"Cluster {remaining_c}: Weekend Reader Archetype"
    
    df_r['segment_name'] = df_r['cluster_kmeans'].map(archetype_map)
    
    return {
        'reader_clustered_df': df_r,
        'kmeans_model': kmeans,
        'scaler': scaler,
        'silhouette_kmeans': round(sil_kmeans, 4),
        'silhouette_agg': round(sil_agg, 4),
        'cluster_profiles': cluster_profiles,
        'archetype_map': archetype_map,
        'feature_cols': valid_cols
    }

def save_clustering_artifacts(clustering_res, model_dir="models"):
    """
    Saves clustering objects to models directory.
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clustering_res['kmeans_model'], os.path.join(model_dir, "kmeans_clusterer.joblib"))
    joblib.dump(clustering_res['scaler'], os.path.join(model_dir, "cluster_scaler.joblib"))
    joblib.dump(clustering_res['archetype_map'], os.path.join(model_dir, "cluster_archetypes.joblib"))
    print(f"[Clustering Engine] Clusterer saved to '{model_dir}'.")

if __name__ == "__main__":
    from src.data_processing import process_and_clean_data
    from src.feature_engineering import calculate_behavioral_features
    
    _, _, _, df_master, _ = process_and_clean_data()
    _, r_feat = calculate_behavioral_features(df_master)
    
    c_res = perform_reader_clustering(r_feat)
    print(f"[Clustering] Completed. Silhouette score (KMeans): {c_res['silhouette_kmeans']}")
    print(c_res['cluster_profiles'])
