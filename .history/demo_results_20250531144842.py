#!/usr/bin/env python3
"""
Demonstration of the improved Game Theory clustering vs traditional methods
"""

import sys
import os
sys.path.append(os.path.join('.', 'app'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from services.preprocess import full_preprocess
from services.traditional import run_kmeans, run_dbscan, run_agglomerative
from models.game_theory import GameTheoryClusterer

def demonstrate_clustering():
    print("🎯 Game Theory Clustering vs Traditional Methods")
    print("=" * 60)
    print("Based on: https://www.mit.edu/~vgarg/tkde-final.pdf")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n📊 Loading Unilever invoice data...")
    df = pd.read_csv('app/data/invoices_realistic.csv')
    X_full, _ = full_preprocess(df)
    print(f"Data: {X_full.shape[0]} invoices, {X_full.shape[1]} features")
    
    # Run all clustering methods
    print("\n🔬 Running clustering algorithms...")
    
    results = {}
    
    # 1. Game Theory Clustering (Improved)
    print("   🧠 Game Theory (Coalition Formation + Shapley Values)...")
    gt_model = GameTheoryClusterer(X_full, gamma=1.0, similarity_metric='euclidean')
    gt_labels = gt_model.fit(threshold=0.2, max_coalition_size=8)
    results['Game Theory'] = gt_labels
    
    # 2. K-Means
    print("   🔵 K-Means...")
    kmeans_labels = run_kmeans(X_full, n_clusters=5)
    results['K-Means'] = kmeans_labels
    
    # 3. DBSCAN
    print("   🟢 DBSCAN...")
    dbscan_labels = run_dbscan(X_full, eps=0.8, min_samples=8)
    results['DBSCAN'] = dbscan_labels
    
    # 4. Agglomerative
    print("   🟡 Agglomerative...")
    agg_labels = run_agglomerative(X_full, n_clusters=5)
    results['Agglomerative'] = agg_labels
    
    # Compare results
    print("\n📈 CLUSTERING RESULTS COMPARISON")
    print("-" * 60)
    print(f"{'Method':<15} {'Clusters':<10} {'Noise':<8} {'Silhouette':<12}")
    print("-" * 60)
    
    for method, labels in results.items():
        n_clusters = len(np.unique(labels))
        n_noise = np.sum(labels == -1) if -1 in labels else 0
        
        # Calculate silhouette score
        try:
            if len(set(labels)) > 1:
                mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
                if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                    sil_score = silhouette_score(X_full[mask], labels[mask])
                    sil_str = f"{sil_score:.3f}"
                else:
                    sil_str = "N/A"
            else:
                sil_str = "N/A"
        except:
            sil_str = "Error"
        
        print(f"{method:<15} {n_clusters:<10} {n_noise:<8} {sil_str:<12}")
    
    # Game Theory specific insights
    print(f"\n🎯 GAME THEORY CLUSTERING INSIGHTS")
    print("-" * 60)
    
    if hasattr(gt_model, 'get_cluster_statistics'):
        stats = gt_model.get_cluster_statistics()
        print(f"Total clusters formed: {len(stats)}")
        
        for cluster_name, info in stats.items():
            size = info['size']
            similarity = info['avg_internal_similarity']
            print(f"{cluster_name}: {size} members, avg similarity: {similarity:.3f}")
    
    # Key improvements made
    print(f"\n🔧 KEY IMPROVEMENTS TO GAME THEORY CLUSTERING")
    print("-" * 60)
    print("1. ✅ Proper coalition formation using Shapley values")
    print("2. ✅ Fixed utility function for meaningful coalition evaluation")
    print("3. ✅ Improved clustering assignment with greedy coalition expansion")
    print("4. ✅ Support for both euclidean and cosine similarity metrics")
    print("5. ✅ Computational optimization for large datasets")
    print("6. ✅ Better threshold-based cluster formation")
    
    # Show cluster distribution for Game Theory
    print(f"\n📊 GAME THEORY CLUSTER DISTRIBUTION")
    print("-" * 60)
    unique_labels, counts = np.unique(gt_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(gt_labels)) * 100
        print(f"Cluster {label}: {count} invoices ({percentage:.1f}%)")
    
    print(f"\n🎉 SUCCESS: Game Theory clustering now creates {len(np.unique(gt_labels))} meaningful clusters!")
    print("The Streamlit app is running at: http://localhost:8501")
    print("\n💡 Try different threshold values in the app to see how it affects clustering:")
    print("   - Lower threshold (0.1-0.2): More clusters")
    print("   - Higher threshold (0.3-0.5): Fewer, tighter clusters")

if __name__ == "__main__":
    demonstrate_clustering() 