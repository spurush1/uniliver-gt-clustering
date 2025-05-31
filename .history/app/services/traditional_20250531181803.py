from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import sparse

def _validate_clustering_input(X):
    """Validate and prepare input data for clustering algorithms."""
    if X is None:
        raise ValueError("Input data cannot be None")
    
    # Ensure dense array
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    
    # Validate
    if X.ndim != 2:
        raise ValueError(f"Input must be 2D array, got {X.ndim}D")
    if X.shape[0] == 0:
        raise ValueError("Input array is empty")
    if X.shape[1] == 0:
        raise ValueError("Input array has no features")
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values")
    
    return X

def run_kmeans(X, n_clusters=3, random_state=42):
    """Run K-Means clustering"""
    X = _validate_clustering_input(X)
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X)

def run_dbscan(X, eps=0.5, min_samples=5):
    """Run DBSCAN clustering"""
    X = _validate_clustering_input(X)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

def run_agglomerative(X, n_clusters=3, linkage='ward'):
    """Run Agglomerative clustering"""
    X = _validate_clustering_input(X)
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X)

def auto_tune_dbscan(X, eps_range=(0.1, 2.0), min_samples_range=(3, 10)):
    """
    Auto-tune DBSCAN parameters to find reasonable clustering
    """
    from sklearn.metrics import silhouette_score
    
    X = _validate_clustering_input(X)
    best_score = -1
    best_params = None
    best_labels = None
    
    eps_values = np.linspace(eps_range[0], eps_range[1], 10)
    min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            
            # Check if we have multiple clusters and not too many noise points
            unique_labels = set(labels)
            n_samples = X.shape[0]
            if len(unique_labels) > 1 and len(unique_labels) < n_samples * 0.8:
                # Filter out noise points for silhouette score
                mask = labels != -1
                if np.sum(mask) > 10 and len(set(labels[mask])) > 1:
                    try:
                        score = silhouette_score(X[mask], labels[mask])
                        if score > best_score:
                            best_score = score
                            best_params = (eps, min_samples)
                            best_labels = labels
                    except:
                        continue
    
    if best_labels is not None:
        return best_labels, best_params
    else:
        # Fallback to default parameters
        return DBSCAN(eps=0.5, min_samples=5).fit_predict(X), (0.5, 5)

def determine_optimal_clusters(X, max_clusters=10):
    """
    Use elbow method and silhouette analysis to determine optimal number of clusters
    """
    from sklearn.metrics import silhouette_score
    
    X = _validate_clustering_input(X)
    inertias = []
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, X.shape[0]))
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Find elbow point (simplified)
    if len(inertias) >= 2:
        # Calculate rate of change
        rates = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        
        # Find the point where improvement slows down significantly
        if len(rates) >= 2:
            improvements = [rates[i] - rates[i+1] for i in range(len(rates)-1)]
            elbow_idx = np.argmax(improvements) + 2  # +2 because we start from 2 clusters
        else:
            elbow_idx = 2
    else:
        elbow_idx = 2
    
    # Best silhouette score
    best_sil_idx = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
    
    return {
        'elbow_method': min(elbow_idx, max_clusters),
        'silhouette_method': min(best_sil_idx, max_clusters),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'cluster_range': list(cluster_range)
    }
