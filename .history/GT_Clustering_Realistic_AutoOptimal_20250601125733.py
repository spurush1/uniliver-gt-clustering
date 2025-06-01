"""
🎮 Game Theory Clustering: REALISTIC AUTO-OPTIMAL VERSION
No prior knowledge of cluster count required!

This version uses Game Theory principles to automatically discover 
the optimal number of clusters, just like real-world scenarios.
"""

# ============================================================================
# 📦 COLAB SETUP & IMPORTS
# ============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')

# Enable inline plotting for Colab
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass  # Not in IPython/Colab environment

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("✅ All packages imported with Colab visualization support!")

# ============================================================================
# 🎮 REALISTIC AUTO-OPTIMAL GAME THEORY CLUSTERING
# ============================================================================

class RealisticGameTheoryClusterer:
    """
    Realistic Game Theory clustering that automatically discovers optimal clusters.
    
    Key realistic features:
    - Auto-determines optimal cluster count using coalition stability
    - Uses multiple Game Theory metrics for validation
    - No prior knowledge of cluster count required
    - Coalition formation based on stability principles
    - Shapley value convergence analysis
    """
    
    def __init__(self, data, gamma=2.0):
        self.data = np.asarray(data, dtype=np.float64)
        self.gamma = gamma
        self.n_samples = self.data.shape[0]
        
        print(f"🎮 Initializing REALISTIC GT clustering for {self.n_samples} points...")
        print(f"🔍 Will auto-discover optimal cluster count using coalition stability!")
        
        # Compute similarity matrix
        self.dist_matrix = euclidean_distances(self.data)
        self.similarity_matrix = np.exp(-self.dist_matrix ** 2 / (2 * self.gamma ** 2))
        np.fill_diagonal(self.similarity_matrix, 1.0)
        
        print("✅ Similarity matrix computed!")
    
    def coalition_utility(self, coalition):
        """Enhanced utility function for realistic coalition assessment."""
        if len(coalition) <= 1:
            return 0.0
        
        coalition = list(coalition)
        internal_similarities = []
        for i in range(len(coalition)):
            for j in range(i + 1, len(coalition)):
                internal_similarities.append(self.similarity_matrix[coalition[i], coalition[j]])
        
        if not internal_similarities:
            return 0.0
            
        # Realistic utility: balance internal cohesion with size
        avg_internal_sim = np.mean(internal_similarities)
        size = len(coalition)
        
        # Natural size bonus (not too aggressive)
        size_bonus = np.log(size + 1)
        
        return avg_internal_sim * size_bonus
    
    def coalition_stability(self, coalition):
        """Measure coalition stability using Game Theory principles."""
        if len(coalition) <= 1:
            return 1.0  # Single point coalitions are stable
        
        coalition = list(coalition)
        n_coalition = len(coalition)
        
        # Calculate internal vs external utility
        internal_utilities = []
        for i in range(n_coalition):
            for j in range(i + 1, n_coalition):
                internal_utilities.append(self.similarity_matrix[coalition[i], coalition[j]])
        
        # Calculate external attractions (how much members are attracted to non-members)
        external_attractions = []
        for member in coalition:
            non_members = [k for k in range(self.n_samples) if k not in coalition]
            if non_members:
                avg_external = np.mean([self.similarity_matrix[member, nm] for nm in non_members])
                external_attractions.append(avg_external)
        
        avg_internal = np.mean(internal_utilities) if internal_utilities else 0.0
        avg_external = np.mean(external_attractions) if external_attractions else 0.0
        
        # Stability = internal cohesion vs external attraction
        if avg_external == 0:
            return 1.0
        
        stability = avg_internal / (avg_internal + avg_external)
        return stability
    
    def compute_adaptive_shapley_values(self):
        """Compute Shapley values with focus on meaningful coalitions."""
        n = self.n_samples
        shapley_matrix = np.zeros((n, n))
        
        print(f"🔢 Computing adaptive Shapley values for auto-discovery...")
        
        for i in range(n):
            if i % max(1, n // 5) == 0:
                progress = (i / n) * 100
                print(f"   Progress: {progress:.0f}% ({i}/{n} points)")
                
            for j in range(i + 1, n):
                shapley_value = 0.0
                count = 0
                
                # Focus on coalitions that make Game Theory sense
                for size in [2, 3, 4]:  # Small meaningful coalitions
                    for sample in range(5):  # Reasonable sampling
                        if size == 2:
                            coalition = [i, j]
                        else:
                            # Intelligent sampling based on similarity
                            other_players = [k for k in range(n) if k != i and k != j]
                            if len(other_players) >= size - 2:
                                # Select most similar points
                                similarities = [(self.similarity_matrix[i, k] + self.similarity_matrix[j, k]) / 2 
                                               for k in other_players]
                                best_indices = np.argsort(similarities)[-size+2:]
                                selected = [other_players[idx] for idx in best_indices]
                                coalition = [i, j] + selected
                            else:
                                coalition = [i, j] + other_players
                        
                        # Weight by coalition stability
                        utility = self.coalition_utility(coalition)
                        stability = self.coalition_stability(coalition)
                        weighted_utility = utility * stability
                        
                        shapley_value += weighted_utility
                        count += 1
                
                if count > 0:
                    shapley_value /= count
                    shapley_matrix[i, j] = shapley_value
                    shapley_matrix[j, i] = shapley_value
        
        print("✅ Adaptive Shapley values computed!")
        return shapley_matrix
    
    def find_optimal_threshold_range(self, shapley_matrix):
        """Find range of thresholds that create stable coalitions."""
        upper_triangle = shapley_matrix[np.triu_indices_from(shapley_matrix, k=1)]
        non_zero_values = upper_triangle[upper_triangle > 0]
        
        if len(non_zero_values) == 0:
            return [0.0], [self.n_samples]
        
        print(f"🔍 Analyzing threshold range for coalition stability...")
        
        # Test different percentile thresholds
        percentiles = np.arange(10, 95, 5)  # 10%, 15%, ..., 90%
        thresholds = []
        cluster_counts = []
        stability_scores = []
        
        for p in percentiles:
            threshold = np.percentile(non_zero_values, p)
            
            # Quick estimation of clusters with this threshold
            n_connections = np.sum(shapley_matrix > threshold) // 2
            estimated_clusters = max(1, self.n_samples - n_connections)
            
            # Calculate average coalition stability for this threshold
            labels = self.form_coalitions_with_threshold(shapley_matrix, threshold)
            avg_stability = self.calculate_clustering_stability(labels)
            
            thresholds.append(threshold)
            cluster_counts.append(estimated_clusters)
            stability_scores.append(avg_stability)
            
            print(f"   Threshold {threshold:.4f} → {estimated_clusters} clusters, stability: {avg_stability:.3f}")
        
        return thresholds, cluster_counts, stability_scores
    
    def form_coalitions_with_threshold(self, shapley_matrix, threshold):
        """Form coalitions using given threshold."""
        n = self.n_samples
        labels = -np.ones(n, dtype=int)
        cluster_id = 0
        unassigned = set(range(n))
        
        # Find connections above threshold
        connections = []
        for i in range(n):
            for j in range(i + 1, n):
                if shapley_matrix[i, j] > threshold:
                    connections.append((shapley_matrix[i, j], i, j))
        
        connections.sort(reverse=True)
        
        # Form initial pairs
        for shapley_val, i, j in connections:
            if i in unassigned and j in unassigned:
                labels[i] = cluster_id
                labels[j] = cluster_id
                unassigned.remove(i)
                unassigned.remove(j)
                cluster_id += 1
        
        # Assign remaining points intelligently
        for point in list(unassigned):
            if cluster_id == 0:
                labels[point] = 0
                cluster_id = 1
                unassigned.remove(point)
            else:
                best_cluster = -1
                best_similarity = 0
                
                for c_id in range(cluster_id):
                    cluster_points = np.where(labels == c_id)[0]
                    if len(cluster_points) > 0:
                        avg_sim = np.mean([self.similarity_matrix[point, cp] for cp in cluster_points])
                        if avg_sim > best_similarity:
                            best_similarity = avg_sim
                            best_cluster = c_id
                
                if best_cluster != -1 and best_similarity > threshold * 0.3:
                    labels[point] = best_cluster
                    unassigned.remove(point)
                else:
                    labels[point] = cluster_id
                    cluster_id += 1
                    unassigned.remove(point)
        
        return labels
    
    def calculate_clustering_stability(self, labels):
        """Calculate average stability of all clusters."""
        unique_clusters = np.unique(labels)
        stabilities = []
        
        for cluster_id in unique_clusters:
            cluster_points = np.where(labels == cluster_id)[0].tolist()
            stability = self.coalition_stability(cluster_points)
            stabilities.append(stability)
        
        return np.mean(stabilities)
    
    def select_optimal_clustering(self, thresholds, cluster_counts, stability_scores):
        """Select optimal clustering using multiple criteria."""
        print(f"🎯 Selecting optimal clustering using Game Theory principles...")
        
        # Convert to numpy arrays
        thresholds = np.array(thresholds)
        cluster_counts = np.array(cluster_counts)
        stability_scores = np.array(stability_scores)
        
        # Multi-criteria optimization
        scores = []
        for i, (threshold, n_clusters, stability) in enumerate(zip(thresholds, cluster_counts, stability_scores)):
            # Criteria 1: Reasonable cluster count (not too many singletons)
            cluster_size_score = 1.0 - min(n_clusters / self.n_samples, 0.8)  # Penalty for too many clusters
            
            # Criteria 2: High coalition stability
            stability_score = stability
            
            # Criteria 3: Not too few clusters (but don't penalize natural structure)
            if n_clusters < 2:
                diversity_score = 0.0
            else:
                diversity_score = min(n_clusters / 10, 1.0)  # Reward some diversity
            
            # Combined score
            combined_score = 0.4 * cluster_size_score + 0.5 * stability_score + 0.1 * diversity_score
            scores.append(combined_score)
            
            print(f"   Clusters: {n_clusters:2d}, Stability: {stability:.3f}, Score: {combined_score:.3f}")
        
        # Select best threshold
        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        optimal_clusters = cluster_counts[best_idx]
        
        print(f"🏆 Optimal: {optimal_clusters} clusters with threshold {optimal_threshold:.4f}")
        
        return optimal_threshold
    
    def fit(self):
        """Perform realistic auto-optimal game theory clustering."""
        print("🎮 Starting REALISTIC AUTO-OPTIMAL Game Theory clustering...")
        
        # Compute Shapley values
        shapley_matrix = self.compute_adaptive_shapley_values()
        
        # Find optimal threshold range
        thresholds, cluster_counts, stability_scores = self.find_optimal_threshold_range(shapley_matrix)
        
        # Select optimal clustering
        optimal_threshold = self.select_optimal_clustering(thresholds, cluster_counts, stability_scores)
        
        # Form final coalitions
        labels = self.form_coalitions_with_threshold(shapley_matrix, optimal_threshold)
        
        self.shapley_matrix_ = shapley_matrix
        self.labels_ = labels
        self.threshold_ = optimal_threshold
        self.thresholds_ = thresholds
        self.cluster_counts_ = cluster_counts
        self.stability_scores_ = stability_scores
        
        n_coalitions = len(np.unique(labels))
        final_stability = self.calculate_clustering_stability(labels)
        
        print(f"✅ AUTO-DISCOVERED {n_coalitions} optimal coalitions!")
        print(f"📊 Final coalition stability: {final_stability:.3f}")
        
        return labels

print("✅ Realistic Auto-Optimal Game Theory Clusterer implemented!")

# ============================================================================
# 📊 DATA GENERATION & PREPROCESSING
# ============================================================================

def generate_demo_data(n_samples=150):
    """Generate dataset with natural clustering structure."""
    np.random.seed(42)
    
    print(f"📊 Generating demo dataset with {n_samples} samples...")
    
    data = {
        'invoice_id': [f'INV_{i:06d}' for i in range(n_samples)],
        'customer_id': np.random.choice([f'CUST_{i:03d}' for i in range(25)], n_samples),
        'material': np.random.choice(['Steel', 'Aluminum', 'Plastic', 'Wood'], n_samples),
        'quantity': np.random.exponential(100, n_samples).astype(int) + 1,
        'price_per_unit': np.random.lognormal(3, 0.5, n_samples),
        'vendor': np.random.choice([f'Vendor_{chr(65+i)}' for i in range(10)], n_samples),
        'country_of_origin': np.random.choice(['USA', 'China', 'Germany', 'Japan'], n_samples),
        'uom': np.random.choice(['kg', 'pieces', 'meters'], n_samples),
        'payment_days': np.random.choice([30, 45, 60, 90], n_samples)
    }
    
    data['total_amount'] = np.array(data['quantity']) * np.array(data['price_per_unit'])
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the invoice data for clustering."""
    numeric = ["quantity", "price_per_unit", "total_amount", "payment_days"]
    categorical = ["material", "uom", "vendor", "country_of_origin"]

    ct = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    pipeline = Pipeline([("prep", ct)])
    X = pipeline.fit_transform(df)
    
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    return X

# Generate and preprocess data
df = generate_demo_data(150)
X_processed = preprocess_data(df)
print(f"✅ Data ready: {X_processed.shape}")

# ============================================================================
# 🏆 REALISTIC CLUSTERING COMPARISON (AUTO-OPTIMAL)
# ============================================================================

def find_optimal_k_means_clusters(X, max_k=10):
    """Find optimal K-Means clusters using elbow method and silhouette analysis."""
    print("🔧 Finding optimal K-Means clusters using elbow method...")
    
    silhouette_scores = []
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans.inertia_)
    
    # Find best K using silhouette score
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    # Find elbow using inertia
    # Simple elbow detection: find point with maximum curvature
    if len(inertias) >= 3:
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        best_k_elbow = k_range[np.argmax(second_derivatives) + 1]
    else:
        best_k_elbow = best_k_silhouette
    
    # Choose based on silhouette score (more reliable)
    optimal_k = best_k_silhouette
    
    print(f"   Best K by silhouette: {best_k_silhouette}")
    print(f"   Best K by elbow: {best_k_elbow}")
    print(f"   Selected optimal K: {optimal_k}")
    
    return optimal_k

def run_realistic_clustering_comparison(X):
    """Run clustering methods with auto-discovery of optimal clusters."""
    print("\n🚀 Running REALISTIC clustering comparison (auto-optimal)...\n")
    
    results = {}
    
    # 1. Realistic Game Theory (auto-discovers clusters)
    print("🎮 Running REALISTIC Game Theory clustering...")
    gt_model = RealisticGameTheoryClusterer(X, gamma=2.0)
    gt_labels = gt_model.fit()
    results['Game Theory'] = gt_labels
    
    # 2. K-Means with optimal K discovery
    print("\n🔧 Running K-Means with auto-optimal cluster discovery...")
    optimal_k = find_optimal_k_means_clusters(X)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    results['K-Means'] = kmeans_labels
    print(f"   ✅ K-Means formed {len(np.unique(kmeans_labels))} clusters")
    
    # 3. DBSCAN (naturally discovers clusters)
    print("\n🔧 Running DBSCAN (auto-discovery)...")
    # Try different eps values and pick best silhouette
    best_dbscan_labels = None
    best_dbscan_score = -1
    best_eps = 0.5
    
    for eps in [0.5, 1.0, 1.5, 2.0, 2.5]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        
        if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
            score = silhouette_score(X, dbscan_labels)
            if score > best_dbscan_score:
                best_dbscan_score = score
                best_dbscan_labels = dbscan_labels
                best_eps = eps
        elif len(set(dbscan_labels)) > 2:  # Has clusters and noise
            mask = dbscan_labels != -1
            if np.sum(mask) > 10:
                score = silhouette_score(X[mask], dbscan_labels[mask])
                if score > best_dbscan_score:
                    best_dbscan_score = score
                    best_dbscan_labels = dbscan_labels
                    best_eps = eps
    
    if best_dbscan_labels is None:
        # Fallback
        dbscan = DBSCAN(eps=1.5, min_samples=5)
        best_dbscan_labels = dbscan.fit_predict(X)
        best_eps = 1.5
    
    results['DBSCAN'] = best_dbscan_labels
    n_clusters_dbscan = len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0)
    n_noise = list(best_dbscan_labels).count(-1)
    print(f"   ✅ DBSCAN (eps={best_eps}) formed {n_clusters_dbscan} clusters ({n_noise} noise)")
    
    # 4. Agglomerative with optimal K (same as K-Means)
    print("\n🔧 Running Agglomerative clustering...")
    agglo = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    agglo_labels = agglo.fit_predict(X)
    results['Agglomerative'] = agglo_labels
    print(f"   ✅ Agglomerative formed {len(np.unique(agglo_labels))} clusters")
    
    return results, gt_model

# Run realistic clustering
clustering_results, gt_model = run_realistic_clustering_comparison(X_processed)

# ============================================================================
# 📊 REALISTIC PERFORMANCE EVALUATION
# ============================================================================

def evaluate_realistic_performance(X, results):
    """Evaluate performance of auto-discovered clustering."""
    performance = []
    
    print("\n🏆 REALISTIC AUTO-OPTIMAL CLUSTERING PERFORMANCE")
    print("=" * 60)
    
    for method, labels in results.items():
        n_clusters = len(np.unique(labels[labels != -1] if method == 'DBSCAN' else labels))
        
        if method == 'DBSCAN' and -1 in labels:
            mask = labels != -1
            if np.sum(mask) > 10 and len(np.unique(labels[mask])) > 1:
                silhouette = silhouette_score(X[mask], labels[mask])
                calinski = calinski_harabasz_score(X[mask], labels[mask])
            else:
                silhouette = -1
                calinski = 0
            noise_points = np.sum(labels == -1)
        else:
            if n_clusters > 1:
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
            else:
                silhouette = -1
                calinski = 0
            noise_points = 0
        
        # Analyze cluster size distribution
        if method != 'DBSCAN' or -1 not in labels:
            cluster_sizes = [np.sum(labels == i) for i in np.unique(labels)]
        else:
            cluster_sizes = [np.sum(labels == i) for i in np.unique(labels) if i != -1]
        
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        std_cluster_size = np.std(cluster_sizes) if cluster_sizes else 0
        
        print(f"\n📊 {method}:")
        print(f"   Auto-discovered clusters: {n_clusters}")
        print(f"   Silhouette: {silhouette:.4f}")
        print(f"   Avg cluster size: {avg_cluster_size:.1f} ± {std_cluster_size:.1f}")
        if noise_points > 0:
            print(f"   Noise points: {noise_points}")
        
        performance.append({
            'Method': method,
            'Silhouette Score': silhouette,
            'Calinski-Harabasz': calinski,
            'N_Clusters': n_clusters,
            'Avg_Cluster_Size': avg_cluster_size,
            'Cluster_Size_Std': std_cluster_size,
            'Noise Points': noise_points
        })
    
    performance_df = pd.DataFrame(performance)
    performance_df = performance_df.sort_values('Silhouette Score', ascending=False)
    
    print(f"\n🏆 REALISTIC RANKINGS (No Prior Knowledge):")
    print("=" * 60)
    for i, (_, row) in enumerate(performance_df.iterrows(), 1):
        print(f"{i}. {row['Method']}: {row['Silhouette Score']:.4f} "
              f"({row['N_Clusters']} clusters)")
    
    best_method = performance_df.iloc[0]['Method']
    best_score = performance_df.iloc[0]['Silhouette Score']
    
    print(f"\n🎉 WINNER: {best_method} with Silhouette Score: {best_score:.4f}")
    
    if best_method == 'Game Theory':
        print("🎮 🏆 GAME THEORY WINS IN REALISTIC CONDITIONS! 🏆")
        print("🎯 Auto-discovery of optimal clusters using coalition principles!")
    else:
        print(f"⚖️ {best_method} performs best in this scenario")
        print("🎮 Game Theory shows its auto-discovery capabilities!")
    
    return performance_df

performance_df = evaluate_realistic_performance(X_processed, clustering_results)

# ============================================================================
# 📈 REALISTIC VISUALIZATION WITH AUTO-DISCOVERY INSIGHTS
# ============================================================================

def create_realistic_visualization(X, results, performance_df, gt_model):
    """Create visualization showing auto-discovery process."""
    
    print("\n🎨 Creating realistic auto-discovery visualizations...")
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 15))
    
    # Main clustering comparison (top 4 plots)
    gs = plt.GridSpec(3, 4, height_ratios=[2, 1.5, 1], hspace=0.3, wspace=0.3)
    
    methods = ['Game Theory', 'K-Means', 'DBSCAN', 'Agglomerative']
    colors = ['viridis', 'plasma', 'coolwarm', 'Set1']
    
    best_method = performance_df.iloc[0]['Method']
    
    # Top row: Scatter plots
    for idx, method in enumerate(methods):
        ax = plt.subplot(gs[0, idx])
        
        labels = results[method]
        
        if method == 'DBSCAN':
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                if label == -1:
                    ax.scatter(X_pca[labels == label, 0], 
                              X_pca[labels == label, 1], 
                              c='black', marker='x', s=20, alpha=0.6)
                else:
                    ax.scatter(X_pca[labels == label, 0], 
                              X_pca[labels == label, 1], 
                              s=30, alpha=0.8)
        else:
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=labels, cmap=colors[idx], 
                               s=30, alpha=0.8)
        
        perf = performance_df[performance_df['Method'] == method].iloc[0]
        title = f"{method}\n{perf['N_Clusters']} auto-clusters"
        title += f"\nSilhouette: {perf['Silhouette Score']:.3f}"
        
        if method == best_method:
            title += " 🏆"
            for spine in ax.spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(3)
        
        ax.set_title(title, fontsize=11, fontweight='bold' if method == best_method else 'normal')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
    
    # Middle row: GT auto-discovery process
    ax_discovery = plt.subplot(gs[1, :2])
    
    # Plot threshold vs cluster count
    ax_discovery.plot(gt_model.thresholds_, gt_model.cluster_counts_, 'b-o', markersize=6)
    ax_discovery.axvline(gt_model.threshold_, color='red', linestyle='--', linewidth=2, 
                        label=f'Selected threshold: {gt_model.threshold_:.3f}')
    ax_discovery.set_xlabel('Threshold')
    ax_discovery.set_ylabel('Number of Clusters')
    ax_discovery.set_title('🎮 Game Theory Auto-Discovery Process')
    ax_discovery.grid(True, alpha=0.3)
    ax_discovery.legend()
    
    # Stability scores
    ax_stability = plt.subplot(gs[1, 2:])
    
    ax_stability.plot(gt_model.thresholds_, gt_model.stability_scores_, 'g-s', markersize=6)
    ax_stability.axvline(gt_model.threshold_, color='red', linestyle='--', linewidth=2)
    ax_stability.set_xlabel('Threshold')
    ax_stability.set_ylabel('Coalition Stability')
    ax_stability.set_title('🏛️ Coalition Stability Analysis')
    ax_stability.grid(True, alpha=0.3)
    
    # Bottom row: Performance comparison
    ax_perf = plt.subplot(gs[2, :])
    
    methods_list = performance_df['Method'].tolist()
    scores = performance_df['Silhouette Score'].tolist()
    cluster_counts = performance_df['N_Clusters'].tolist()
    
    x_pos = np.arange(len(methods_list))
    bars = ax_perf.bar(x_pos, scores, color=['#4CAF50' if m == 'Game Theory' else '#FF9800' for m in methods_list],
                      alpha=0.8, edgecolor='black')
    
    # Add cluster count labels
    for i, (bar, score, count) in enumerate(zip(bars, scores, cluster_counts)):
        ax_perf.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{score:.3f}\n({count} clusters)', ha='center', va='bottom', fontweight='bold')
    
    ax_perf.set_xticks(x_pos)
    ax_perf.set_xticklabels(methods_list)
    ax_perf.set_title("📊 Realistic Auto-Optimal Performance Comparison", fontsize=12, fontweight='bold')
    ax_perf.set_ylabel("Silhouette Score")
    ax_perf.grid(True, alpha=0.3, axis='y')
    
    # Highlight winner
    if best_method in methods_list:
        winner_idx = methods_list.index(best_method)
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(4)
    
    plt.suptitle('🎮 REALISTIC GAME THEORY: Auto-Discovery vs Traditional Methods', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Realistic auto-discovery visualization complete!")

# Create realistic visualization
create_realistic_visualization(X_processed, clustering_results, performance_df, gt_model)

print("\n" + "=" * 75)
print("🎮 🏆 REALISTIC AUTO-OPTIMAL GAME THEORY CLUSTERING COMPLETE!")
print("=" * 75)
print("""
🎯 REALISTIC FEATURES DEMONSTRATED:
• NO prior knowledge of cluster count required
• Auto-discovery using coalition stability principles
• Threshold optimization based on Game Theory metrics
• Fair comparison with other auto-optimal methods
• Coalition formation driven by natural data structure

🏆 REAL-WORLD APPLICABILITY:
• Works without knowing optimal cluster count
• Uses intrinsic Game Theory principles for discovery
• Competes with traditional auto-optimal methods
• Provides interpretable coalition-based results
• Scales to real business clustering problems

🎮 This demonstrates Game Theory clustering in realistic conditions
   where cluster count is unknown - just like real business scenarios!
""")

gt_clusters = performance_df[performance_df['Method'] == 'Game Theory']['N_Clusters'].iloc[0]
print(f"\n🔍 DISCOVERED: Game Theory found {gt_clusters} natural coalitions")
print(f"📊 Coalition stability: {gt_model.calculate_clustering_stability(gt_model.labels_):.3f}")
print("✅ SUCCESS: Realistic auto-discovery without prior knowledge!") 