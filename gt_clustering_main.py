"""
🎯 GT Clustering Main Analysis
Fast GT clustering analysis for Excel data with strategic coalition formation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import warnings
warnings.filterwarnings('ignore')

class GTClusterer:
    """Fast GT clustering implementation with Shapley value optimization."""
    
    def __init__(self, data, target_clusters=None):
        self.data = data
        self.n_samples = len(data)
        self.target_clusters = target_clusters or max(5, min(20, len(data) // 50))
        
        # Compute similarity matrix using game theory principles
        print(f"🎮 Computing strategic similarity matrix...")
        distances = euclidean_distances(data)
        max_dist = np.max(distances)
        
        if max_dist > 0:
            # Normalize distances
            distances = distances / max_dist
        
        # Exponential similarity (Gaussian kernel)
        gamma = 0.5  # Tunable parameter
        self.similarity = np.exp(-distances ** 2 / (2 * gamma ** 2))
        np.fill_diagonal(self.similarity, 1.0)
        
        print(f"🎯 Target coalitions: {self.target_clusters}")
    
    def compute_shapley_values(self):
        """Compute Shapley values for each entity."""
        print("🧮 Computing Shapley values...")
        shapley_values = np.zeros(self.n_samples)
        
        # Simplified Shapley value: average contribution to all coalitions
        for i in range(self.n_samples):
            shapley_values[i] = np.mean(self.similarity[i, :])
        
        return shapley_values
    
    def form_strategic_coalitions(self):
        """Form strategic coalitions using game theory principles."""
        print("🤝 Forming strategic coalitions...")
        
        # Initialize each entity as its own coalition
        labels = np.arange(self.n_samples)
        
        # Strategic threshold (top 5% similarities)
        threshold = np.percentile(self.similarity.flatten(), 95)
        print(f"📊 Strategic threshold: {threshold:.4f}")
        
        # Find strategic connections
        connections = []
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                if self.similarity[i, j] > threshold:
                    connections.append((self.similarity[i, j], i, j))
        
        # Sort by strategic value (highest first)
        connections.sort(reverse=True)
        print(f"🔗 Found {len(connections)} strategic connections")
        
        # Maximum coalition size to prevent monopolies
        max_coalition_size = max(10, self.n_samples // self.target_clusters)
        
        # Form coalitions greedily
        for strength, i, j in connections:
            if labels[i] != labels[j]:  # Different coalitions
                # Check coalition sizes
                size_i = np.sum(labels == labels[i])
                size_j = np.sum(labels == labels[j])
                
                if size_i + size_j <= max_coalition_size:
                    # Merge coalitions
                    old_label = labels[j]
                    new_label = labels[i]
                    labels[labels == old_label] = new_label
        
        # Handle singletons (entities not in any meaningful coalition)
        unique_labels = np.unique(labels)
        for point in range(self.n_samples):
            if np.sum(labels == labels[point]) == 1:  # Singleton
                # Find best coalition to join
                best_coalition = -1
                best_score = -1
                
                for coalition_id in unique_labels:
                    if coalition_id == labels[point]:
                        continue
                    
                    coalition_members = np.where(labels == coalition_id)[0]
                    coalition_size = len(coalition_members)
                    
                    if coalition_size >= max_coalition_size:
                        continue
                    
                    # Calculate strategic fit
                    affinities = [self.similarity[point, member] for member in coalition_members]
                    avg_affinity = np.mean(affinities)
                    
                    # Size penalty to prefer smaller coalitions
                    size_bonus = 1.0 / (1 + coalition_size / max_coalition_size)
                    strategic_score = avg_affinity * size_bonus
                    
                    if strategic_score > best_score:
                        best_score = strategic_score
                        best_coalition = coalition_id
                
                # Join best coalition if good enough
                if best_coalition != -1 and best_score > 0.1:
                    labels[point] = best_coalition
        
        # Relabel coalitions sequentially
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
        
        return labels
    
    def analyze_results(self, labels):
        """Analyze GT clustering results."""
        n_clusters = len(np.unique(labels))
        silhouette = silhouette_score(self.data, labels)
        
        coalition_sizes = [np.sum(labels == i) for i in np.unique(labels)]
        
        results = {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'coalition_sizes': coalition_sizes,
            'avg_coalition_size': np.mean(coalition_sizes),
            'max_coalition_size': max(coalition_sizes),
            'min_coalition_size': min(coalition_sizes),
            'total_entities': len(labels)
        }
        
        return results

def run_traditional_clustering_comparison(X):
    """Run traditional clustering methods for comparison."""
    print("🔄 Running traditional clustering comparison...")
    
    methods = {}
    
    # K-Means variants
    for k in [5, 8, 10, 15, 20]:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            sil_score = silhouette_score(X, labels)
            methods[f'KMeans_k{k}'] = {
                'labels': labels,
                'silhouette': sil_score,
                'n_clusters': k,
                'method': 'KMeans'
            }
        except:
            pass
    
    # Agglomerative Clustering
    for k in [5, 8, 10, 15]:
        try:
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = agg.fit_predict(X)
            sil_score = silhouette_score(X, labels)
            methods[f'Agglomerative_k{k}'] = {
                'labels': labels,
                'silhouette': sil_score,
                'n_clusters': k,
                'method': 'Agglomerative'
            }
        except:
            pass
    
    # DBSCAN
    for eps in [0.3, 0.5, 0.7, 1.0]:
        try:
            dbscan = DBSCAN(eps=eps, min_samples=3)
            labels = dbscan.fit_predict(X)
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                methods[f'DBSCAN_eps{eps}'] = {
                    'labels': labels,
                    'silhouette': sil_score,
                    'n_clusters': len(np.unique(labels)),
                    'method': 'DBSCAN'
                }
        except:
            pass
    
    return methods

def load_and_preprocess_data(filepath):
    """Load and preprocess the Excel data."""
    print(f"📁 Loading data from: {filepath}")
    
    try:
        df = pd.read_excel(filepath)
        print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None
    
    # Use first 8 columns as features (adjust as needed)
    feature_cols = df.columns[:8].tolist()
    print(f"📊 Using features: {feature_cols}")
    
    features = df[feature_cols].copy()
    
    # Handle categorical variables
    for col in features.columns:
        if features[col].dtype == 'object':
            print(f"🔧 Encoding categorical column: {col}")
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
    
    # Handle missing values
    features = features.fillna(features.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    print(f"🔧 Preprocessed data shape: {X.shape}")
    
    return df, X

def save_results(df, gt_labels, gt_results, traditional_results, filename_prefix="GT_Analysis"):
    """Save GT clustering results to Excel."""
    print("💾 Saving results...")
    
    # Add GT results to original dataframe
    results_df = df.copy()
    results_df['GT_Strategic_Coalition'] = gt_labels
    results_df['GT_Coalition_Size'] = [np.sum(gt_labels == label) for label in gt_labels]
    
    # Save to Excel
    excel_filename = f"{filename_prefix}_Results.xlsx"
    results_df.to_excel(excel_filename, index=False)
    
    print(f"✅ Results saved to: {excel_filename}")
    
    # Print summary
    print("\n📊 GT CLUSTERING SUMMARY")
    print("=" * 50)
    print(f"🎯 Strategic Coalitions: {gt_results['n_clusters']}")
    print(f"📈 Silhouette Score: {gt_results['silhouette_score']:.4f}")
    print(f"📊 Average Coalition Size: {gt_results['avg_coalition_size']:.1f}")
    print(f"📏 Coalition Size Range: {gt_results['min_coalition_size']}-{gt_results['max_coalition_size']}")
    
    if traditional_results:
        best_traditional = max(traditional_results.items(), key=lambda x: x[1]['silhouette'])
        best_name, best_result = best_traditional
        improvement = ((gt_results['silhouette_score'] - best_result['silhouette']) / abs(best_result['silhouette'])) * 100
        
        print(f"\n🏆 PERFORMANCE COMPARISON")
        print(f"📊 GT vs Best Traditional ({best_name}):")
        print(f"   GT Silhouette: {gt_results['silhouette_score']:.4f}")
        print(f"   Traditional: {best_result['silhouette']:.4f}")
        print(f"   Improvement: {improvement:+.1f}%")
    
    return excel_filename

def main():
    """Main GT clustering analysis."""
    print("🎯 GAME THEORY CLUSTERING ANALYSIS")
    print("=" * 50)
    
    # Load data
    filepath = 'data/clustering_results_named_clusters_with_labels (1).xlsx'
    df, X = load_and_preprocess_data(filepath)
    
    if df is None:
        print("❌ Failed to load data. Please check the file path.")
        return
    
    # Run GT clustering
    print("\n🎮 GAME THEORY CLUSTERING")
    print("-" * 30)
    gt_clusterer = GTClusterer(X, target_clusters=16)  # Adjust target as needed
    gt_labels = gt_clusterer.form_strategic_coalitions()
    gt_results = gt_clusterer.analyze_results(gt_labels)
    
    # Run traditional clustering for comparison
    print("\n🔄 TRADITIONAL CLUSTERING COMPARISON")
    print("-" * 40)
    traditional_results = run_traditional_clustering_comparison(X)
    
    # Save results
    print("\n💾 SAVING RESULTS")
    print("-" * 20)
    excel_filename = save_results(df, gt_labels, gt_results, traditional_results)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {excel_filename}")
    print("🎯 Ready for strategic decision making!")

if __name__ == "__main__":
    main() 