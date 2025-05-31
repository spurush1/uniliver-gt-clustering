
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def run_kmeans(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(X)

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def run_agglomerative(X, n_clusters=3):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X)
