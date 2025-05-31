
from sklearn.cluster import KMeans

def run_kmeans(X, n_clusters=3):
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
