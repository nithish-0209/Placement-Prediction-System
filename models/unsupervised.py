"""
M4: Unsupervised Learning
Implements and analyzes clustering, dimensionality reduction, and anomaly detection:
- K-Means (Lloyd's, K-Means++, Mini-Batch), Elbow Method (WCSS/Inertia), Silhouette Score
- Hierarchical Agglomerative Clustering (Ward, Complete, Average Linkage)
- Density-Based Clustering (DBSCAN: eps, minPts, non-convex geometry, noise detection)
- Principal Component Analysis (PCA: Eigen-decomposition, Standardisation Pitfall)
- Anomaly Detection with Isolation Forest
- t-SNE & UMAP caveats & practical guidelines
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

def compute_kmeans_elbow_silhouette(X, k_range=range(2, 8), sample_size=4000, random_state=42):
    """
    Computes Elbow curve (Inertia / WCSS) and Silhouette Scores
    across multiple values of K to determine optimal cluster count.
    Uses MiniBatchKMeans for computational efficiency on 50k dataset.
    """
    X_sub = X[:sample_size]
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        mbk = MiniBatchKMeans(n_clusters=k, batch_size=512, random_state=random_state, n_init=3)
        labels = mbk.fit_predict(X_sub)
        inertias.append(float(mbk.inertia_))
        sil = silhouette_score(X_sub, labels, sample_size=1500, random_state=random_state)
        silhouette_scores.append(float(sil))

    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }

def fit_dbscan(X, eps=2.2, min_samples=15, max_samples=3000):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
    - eps: neighborhood radius
    - min_samples (minPts): core point threshold
    - Naturally discovers non-convex shaped clusters
    - Flags noise points as cluster -1 (inherent outlier detection)
    """
    X_sub = X[:max_samples]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_sub)
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))
    noise_pct = float(noise_count / len(labels) * 100)

    return {
        'n_clusters': n_clusters,
        'noise_count': noise_count,
        'noise_percentage': round(noise_pct, 2),
        'unique_labels': [int(l) for l in sorted(unique_labels)]
    }

def fit_pca(X, n_components=4):
    """
    Principal Component Analysis (PCA):
    Orthogonal projection maximizing variance.
    Pitfall: If features are not standardized first, features with larger arbitrary
    scales (like IQ: 80-140) dominate components regardless of information content!
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    
    exp_var = [float(v) for v in pca.explained_variance_ratio_]
    cum_var = [float(c) for c in np.cumsum(pca.explained_variance_ratio_)]

    return {
        'pca_model': pca,
        'explained_variance_ratio': exp_var,
        'cumulative_variance_ratio': cum_var,
        'n_components': n_components
    }

def fit_isolation_forest(X, contamination=0.03, random_state=42):
    """
    Isolation Forest for Unsupervised Anomaly Detection:
    Isolates anomalies instead of profiling normal points.
    Anomalies require fewer random axis-aligned splits to be isolated in tree depth.
    Score < 0 (label -1) denotes anomalous profile.
    """
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(X)
    return iso

UNSUPERVISED_CONCEPTS = {
    "hierarchical_linkages": [
        {"name": "Ward", "description": "Minimizes the variance of clusters being merged. Tends to produce equal-sized, spherical clusters."},
        {"name": "Complete", "description": "Considers maximum distance between points in two clusters. Avoids chaining, produces compact clusters."},
        {"name": "Average", "description": "Considers average distance between all pairs. Robust compromise between single and complete."},
        {"name": "Single", "description": "Considers minimum distance between any pair. Susceptible to chaining phenomenon."}
    ],
    "tsne_umap_pitfalls": [
        {"rule": "Local vs Global Distances", "explanation": "t-SNE preserves local neighbor relationships only. The distance between separated distant clusters in t-SNE plot is meaningless and cannot be used to infer global similarity."},
        {"rule": "Perplexity Sensitivity", "explanation": "Changing perplexity (5 vs 30 vs 50) drastically changes visual topology, creating artificial clusters from uniform noise."},
        {"rule": "UMAP Speed & Topology", "explanation": "UMAP preserves more global structure using fuzzy simplicial sets and runs significantly faster than t-SNE, but still distorts metric density."}
    ]
}
