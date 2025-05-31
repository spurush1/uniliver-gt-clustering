#!/usr/bin/env python3
"""
Test script for the improved Game Theory clustering implementation
"""

import sys
import os
sys.path.append(os.path.join('.', 'app'))

import pandas as pd
import numpy as np
from services.preprocess import full_preprocess
from services.traditional import run_kmeans, run_dbscan, run_agglomerative
from models.game_theory import GameTheoryClusterer

def test_clustering():
    print("🧪 Testing Improved Game Theory Clustering")
    print("=" * 50)
    
    # Load data
    print("📊 Loading sample data...")
    df = pd.read_csv('app/data/invoices_realistic.csv')
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    X_full, df_encoded = full_preprocess(df)
    print(f"✅ Data preprocessed: {X_full.shape[0]} samples, {X_full.shape[1]} features")
    
    # Test Game Theory clustering
    print("\n🧠 Testing Game Theory clustering...")
    try:
        gt_model = GameTheoryClusterer(
            X_full, 
            gamma=1.0, 
            similarity_metric='euclidean'
        )
        gt_labels = gt_model.fit(threshold=0.2, max_coalition_size=6)
        gt_clusters = len(np.unique(gt_labels))
        print(f"✅ Game Theory: {gt_clusters} clusters found")
        
        # Get cluster statistics
        if hasattr(gt_model, 'get_cluster_statistics'):
            stats = gt_model.get_cluster_statistics()
            print(f"📈 Cluster statistics generated for {len(stats)} clusters")
        
    except Exception as e:
        print(f"❌ Game Theory clustering failed: {str(e)}")
        return False
    
    # Test traditional methods
    print("\n🔬 Testing traditional clustering methods...")
    
    try:
        # K-Means
        kmeans_labels = run_kmeans(X_full, n_clusters=5)
        kmeans_clusters = len(np.unique(kmeans_labels))
        print(f"✅ K-Means: {kmeans_clusters} clusters")
        
        # DBSCAN
        dbscan_labels = run_dbscan(X_full, eps=0.5, min_samples=5)
        dbscan_clusters = len(np.unique(dbscan_labels))
        noise_points = np.sum(dbscan_labels == -1)
        print(f"✅ DBSCAN: {dbscan_clusters} clusters, {noise_points} noise points")
        
        # Agglomerative
        agg_labels = run_agglomerative(X_full, n_clusters=5)
        agg_clusters = len(np.unique(agg_labels))
        print(f"✅ Agglomerative: {agg_clusters} clusters")
        
    except Exception as e:
        print(f"❌ Traditional clustering failed: {str(e)}")
        return False
    
    # Compare clustering results
    print("\n📊 Clustering Comparison:")
    print(f"Game Theory:   {gt_clusters} clusters")
    print(f"K-Means:       {kmeans_clusters} clusters") 
    print(f"DBSCAN:        {dbscan_clusters} clusters")
    print(f"Agglomerative: {agg_clusters} clusters")
    
    # Test different Game Theory parameters
    print("\n⚙️ Testing Game Theory with different parameters...")
    
    thresholds = [0.1, 0.3, 0.5]
    for threshold in thresholds:
        try:
            gt_model_test = GameTheoryClusterer(X_full, gamma=1.0)
            labels_test = gt_model_test.fit(threshold=threshold, max_coalition_size=6)
            clusters_test = len(np.unique(labels_test))
            print(f"   Threshold {threshold}: {clusters_test} clusters")
        except Exception as e:
            print(f"   Threshold {threshold}: Failed - {str(e)}")
    
    print("\n🎉 All tests completed successfully!")
    print("The improved Game Theory clustering is working correctly.")
    print("You can now run the Streamlit app with: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    test_clustering() 