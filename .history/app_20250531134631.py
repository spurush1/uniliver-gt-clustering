import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from services.preprocess import prepare_features, full_preprocess
from services.traditional import run_kmeans, run_dbscan, run_agglomerative, auto_tune_dbscan, determine_optimal_clusters
from models.game_theory import GameTheoryClusterer

# Page configuration
st.set_page_config(
    page_title="🧠 Unilever Invoice Clustering with Game Theory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #d6d9dc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cluster-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧠 Unilever Invoice Clustering: Game Theory vs Traditional Methods")

st.markdown("""
This application compares **Game Theory-based Clustering** with traditional clustering methods on Unilever invoice data.
The Game Theory approach uses **coalition formation** and **Shapley values** to identify natural groupings in the data.

**Inspired by:** [Game Theory based Clustering paper](https://www.mit.edu/~vgarg/tkde-final.pdf)
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Invoice CSV", 
    type=["csv"],
    help="Upload a CSV file with invoice data or use the sample data"
)

# Use sample data if no file uploaded
if uploaded_file is None:
    st.sidebar.info("Using sample data from app/data/invoices_realistic.csv")
    try:
        df = pd.read_csv("app/data/invoices_realistic.csv")
    except FileNotFoundError:
        st.error("Sample data file not found. Please upload a CSV file.")
        st.stop()
else:
    df = pd.read_csv(uploaded_file)

# Data preprocessing
if df is not None:
    st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Show data preview
    with st.expander("📊 Data Preview", expanded=False):
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10))
        
        st.write("**Data Info:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Invoices", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Unique Customers", df['customer_id'].nunique() if 'customer_id' in df.columns else "N/A")

    # Preprocess data
    try:
        X_full, df_encoded = full_preprocess(df)
        st.sidebar.success(f"✅ Data preprocessed: {X_full.shape[1]} features")
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        st.stop()

    # Clustering configuration
    st.sidebar.subheader("🎛️ Clustering Parameters")
    
    # Game Theory parameters
    gt_threshold = st.sidebar.slider(
        "Game Theory Threshold", 
        min_value=0.05, 
        max_value=0.8, 
        value=0.2, 
        step=0.05,
        help="Lower values create more clusters"
    )
    
    gt_similarity = st.sidebar.selectbox(
        "Similarity Metric",
        ["euclidean", "cosine"],
        help="Metric for computing similarities between data points"
    )
    
    gt_gamma = st.sidebar.slider(
        "Gamma (Temperature)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Controls the spread of similarity function"
    )
    
    # Traditional clustering parameters
    n_clusters_traditional = st.sidebar.slider(
        "Number of Clusters (K-Means/Agglomerative)",
        min_value=2,
        max_value=min(20, X_full.shape[0]//5),
        value=5
    )
    
    auto_tune = st.sidebar.checkbox(
        "Auto-tune Parameters",
        value=True,
        help="Automatically find optimal parameters for traditional methods"
    )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Individual Analysis", 
        "⚖️ Method Comparison", 
        "📈 Cluster Insights",
        "📋 Cluster Details"
    ])

    with tab1:
        st.header("🔬 Individual Clustering Analysis")
        
        method = st.selectbox(
            "Select Clustering Method",
            ["Game Theory", "K-Means", "DBSCAN", "Agglomerative"],
            key="individual_method"
        )
        
        if st.button("🚀 Run Clustering", key="run_individual"):
            with st.spinner(f"Running {method} clustering..."):
                
                if method == "Game Theory":
                    model = GameTheoryClusterer(
                        X_full, 
                        gamma=gt_gamma, 
                        similarity_metric=gt_similarity
                    )
                    labels = model.fit(threshold=gt_threshold, max_coalition_size=8)
                    
                elif method == "K-Means":
                    if auto_tune:
                        optimal_info = determine_optimal_clusters(X_full)
                        optimal_k = optimal_info['silhouette_method']
                        st.info(f"Auto-selected {optimal_k} clusters based on silhouette analysis")
                        labels = run_kmeans(X_full, n_clusters=optimal_k)
                    else:
                        labels = run_kmeans(X_full, n_clusters=n_clusters_traditional)
                        
                elif method == "DBSCAN":
                    if auto_tune:
                        labels, params = auto_tune_dbscan(X_full)
                        st.info(f"Auto-tuned DBSCAN: eps={params[0]:.3f}, min_samples={params[1]}")
                    else:
                        labels = run_dbscan(X_full)
                        
                elif method == "Agglomerative":
                    if auto_tune:
                        optimal_info = determine_optimal_clusters(X_full)
                        optimal_k = optimal_info['silhouette_method']
                        st.info(f"Auto-selected {optimal_k} clusters based on silhouette analysis")
                        labels = run_agglomerative(X_full, n_clusters=optimal_k)
                    else:
                        labels = run_agglomerative(X_full, n_clusters=n_clusters_traditional)
                
                # Results
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels)
                n_noise = np.sum(labels == -1) if -1 in labels else 0
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Clusters Found", n_clusters)
                with col2:
                    st.metric("Noise Points", n_noise)
                with col3:
                    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                        mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
                        if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                            sil_score = silhouette_score(X_full[mask], labels[mask])
                            st.metric("Silhouette Score", f"{sil_score:.3f}")
                        else:
                            st.metric("Silhouette Score", "N/A")
                    else:
                        st.metric("Silhouette Score", "N/A")
                with col4:
                    if len(set(labels)) > 1:
                        ch_score = calinski_harabasz_score(X_full, labels)
                        st.metric("Calinski-Harabasz", f"{ch_score:.1f}")
                    else:
                        st.metric("Calinski-Harabasz", "N/A")

                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 PCA Visualization")
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_full)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
                    ax.set_title(f"{method} Clustering (PCA)", fontsize=14, fontweight='bold')
                    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
                    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
                    plt.colorbar(scatter)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📈 Cluster Size Distribution")
                    cluster_counts = pd.Series(labels).value_counts().sort_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = ax.bar(range(len(cluster_counts)), cluster_counts.values, 
                                 color=plt.cm.tab10(np.arange(len(cluster_counts))))
                    ax.set_title("Cluster Size Distribution", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Cluster ID")
                    ax.set_ylabel("Number of Points")
                    ax.set_xticks(range(len(cluster_counts)))
                    ax.set_xticklabels([f"C{i}" if i != -1 else "Noise" for i in cluster_counts.index])
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
                    
                    st.pyplot(fig)

    with tab2:
        st.header("⚖️ Clustering Method Comparison")
        
        if st.button("🔄 Compare All Methods", key="compare_all"):
            with st.spinner("Running all clustering methods..."):
                
                # Run all clustering methods
                results = {}
                
                # Game Theory
                gt_model = GameTheoryClusterer(X_full, gamma=gt_gamma, similarity_metric=gt_similarity)
                gt_labels = gt_model.fit(threshold=gt_threshold, max_coalition_size=8)
                results['Game Theory'] = gt_labels
                
                # K-Means
                if auto_tune:
                    optimal_info = determine_optimal_clusters(X_full)
                    kmeans_labels = run_kmeans(X_full, n_clusters=optimal_info['silhouette_method'])
                else:
                    kmeans_labels = run_kmeans(X_full, n_clusters=n_clusters_traditional)
                results['K-Means'] = kmeans_labels
                
                # DBSCAN
                if auto_tune:
                    dbscan_labels, _ = auto_tune_dbscan(X_full)
                else:
                    dbscan_labels = run_dbscan(X_full)
                results['DBSCAN'] = dbscan_labels
                
                # Agglomerative
                if auto_tune:
                    agg_labels = run_agglomerative(X_full, n_clusters=optimal_info['silhouette_method'])
                else:
                    agg_labels = run_agglomerative(X_full, n_clusters=n_clusters_traditional)
                results['Agglomerative'] = agg_labels
                
                # Compute metrics
                metrics_df = []
                for method, labels in results.items():
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels)
                    n_noise = np.sum(labels == -1) if -1 in labels else 0
                    
                    # Silhouette score
                    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                        mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
                        if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                            sil_score = silhouette_score(X_full[mask], labels[mask])
                        else:
                            sil_score = np.nan
                    else:
                        sil_score = np.nan
                    
                    # Calinski-Harabasz score
                    if len(set(labels)) > 1:
                        ch_score = calinski_harabasz_score(X_full, labels)
                    else:
                        ch_score = np.nan
                    
                    metrics_df.append({
                        'Method': method,
                        'Clusters': n_clusters,
                        'Noise Points': n_noise,
                        'Silhouette Score': sil_score,
                        'Calinski-Harabasz': ch_score
                    })
                
                metrics_df = pd.DataFrame(metrics_df)
                
                # Display comparison table
                st.subheader("📊 Performance Comparison")
                st.dataframe(
                    metrics_df.style.format({
                        'Silhouette Score': '{:.3f}',
                        'Calinski-Harabasz': '{:.1f}'
                    }).highlight_max(axis=0, subset=['Silhouette Score', 'Calinski-Harabasz'])
                )
                
                # Visualization comparison
                st.subheader("🎨 Visual Comparison")
                
                # PCA for visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_full)
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.ravel()
                
                for i, (method, labels) in enumerate(results.items()):
                    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], 
                                            c=labels, cmap='tab10', alpha=0.7, s=30)
                    axes[i].set_title(f"{method}\n({len(np.unique(labels))} clusters)", 
                                    fontsize=12, fontweight='bold')
                    axes[i].set_xlabel("PC1")
                    axes[i].set_ylabel("PC2")
                
                plt.tight_layout()
                st.pyplot(fig)

    with tab3:
        st.header("📈 Cluster Analysis & Insights")
        
        method_for_analysis = st.selectbox(
            "Select Method for Detailed Analysis",
            ["Game Theory", "K-Means", "DBSCAN", "Agglomerative"],
            key="analysis_method"
        )
        
        if st.button("🔍 Analyze Clusters", key="analyze_clusters"):
            with st.spinner(f"Analyzing {method_for_analysis} clusters..."):
                
                # Run selected method
                if method_for_analysis == "Game Theory":
                    model = GameTheoryClusterer(X_full, gamma=gt_gamma, similarity_metric=gt_similarity)
                    labels = model.fit(threshold=gt_threshold, max_coalition_size=8)
                elif method_for_analysis == "K-Means":
                    labels = run_kmeans(X_full, n_clusters=n_clusters_traditional)
                elif method_for_analysis == "DBSCAN":
                    labels = run_dbscan(X_full)
                else:  # Agglomerative
                    labels = run_agglomerative(X_full, n_clusters=n_clusters_traditional)
                
                # Add cluster labels to original dataframe
                df_with_clusters = df.copy()
                df_with_clusters['cluster'] = labels
                
                # Cluster insights
                st.subheader("🏷️ Cluster Characteristics")
                
                # Numerical features analysis
                numeric_cols = ["quantity", "price_per_unit", "total_amount", "payment_days"]
                available_numeric = [col for col in numeric_cols if col in df.columns]
                
                if available_numeric:
                    cluster_stats = df_with_clusters.groupby('cluster')[available_numeric].agg(['mean', 'std', 'count'])
                    
                    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
                        if cluster_id == -1:
                            st.subheader(f"🔸 Noise Points (n={np.sum(labels == -1)})")
                        else:
                            cluster_size = np.sum(labels == cluster_id)
                            st.subheader(f"🔹 Cluster {cluster_id} (n={cluster_size})")
                        
                        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                        
                        # Show key statistics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if len(available_numeric) >= 2:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                cluster_data[available_numeric[:2]].plot(kind='scatter', 
                                                                        x=available_numeric[0], 
                                                                        y=available_numeric[1], 
                                                                        ax=ax, alpha=0.7)
                                ax.set_title(f"Cluster {cluster_id}: {available_numeric[0]} vs {available_numeric[1]}")
                                st.pyplot(fig)
                        
                        with col2:
                            # Categorical distribution
                            if 'material' in df.columns:
                                material_dist = cluster_data['material'].value_counts().head(5)
                                st.write("**Top Materials:**")
                                st.bar_chart(material_dist)
                
                # Geographic distribution if available
                if 'country_of_origin' in df.columns:
                    st.subheader("🌍 Geographic Distribution")
                    cluster_geo = df_with_clusters.groupby(['cluster', 'country_of_origin']).size().unstack(fill_value=0)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    cluster_geo.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title("Cluster Distribution by Country of Origin")
                    ax.set_xlabel("Cluster")
                    ax.set_ylabel("Number of Invoices")
                    plt.xticks(rotation=0)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    st.pyplot(fig)

    with tab4:
        st.header("📋 Detailed Cluster Information")
        
        method_for_details = st.selectbox(
            "Select Method for Details",
            ["Game Theory", "K-Means", "DBSCAN", "Agglomerative"],
            key="details_method"
        )
        
        if st.button("📊 Show Details", key="show_details"):
            with st.spinner("Generating cluster details..."):
                
                # Run selected method
                if method_for_details == "Game Theory":
                    model = GameTheoryClusterer(X_full, gamma=gt_gamma, similarity_metric=gt_similarity)
                    labels = model.fit(threshold=gt_threshold, max_coalition_size=8)
                    
                    # Show Game Theory specific details
                    if hasattr(model, 'get_cluster_statistics'):
                        st.subheader("🎯 Game Theory Cluster Statistics")
                        stats = model.get_cluster_statistics()
                        
                        for cluster_name, info in stats.items():
                            with st.expander(f"{cluster_name} (Size: {info['size']})"):
                                st.write(f"**Internal Similarity:** {info['avg_internal_similarity']:.3f}")
                                st.write(f"**Members:** {info['members'][:10]}{'...' if len(info['members']) > 10 else ''}")
                
                else:
                    if method_for_details == "K-Means":
                        labels = run_kmeans(X_full, n_clusters=n_clusters_traditional)
                    elif method_for_details == "DBSCAN":
                        labels = run_dbscan(X_full)
                    else:  # Agglomerative
                        labels = run_agglomerative(X_full, n_clusters=n_clusters_traditional)
                
                # Add to dataframe and show downloadable results
                df_result = df.copy()
                df_result['cluster'] = labels
                
                st.subheader("💾 Clustered Data")
                st.dataframe(df_result.head(20))
                
                # Download button
                csv = df_result.to_csv(index=False)
                st.download_button(
                    label="📥 Download Clustered Data (CSV)",
                    data=csv,
                    file_name=f"clustered_invoices_{method_for_details.lower()}.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                st.subheader("📊 Cluster Summary")
                cluster_summary = df_result.groupby('cluster').agg({
                    'total_amount': ['count', 'mean', 'sum'],
                    'quantity': 'mean',
                    'payment_days': 'mean'
                }).round(2)
                
                st.dataframe(cluster_summary)

else:
    st.info("👆 Please upload a CSV file or use the sample data to get started!")

# Footer
st.markdown("---")
st.markdown("""
**📚 About this Application:**
- **Game Theory Clustering** uses coalition formation and Shapley values from cooperative game theory
- **Traditional Methods** include K-Means, DBSCAN, and Agglomerative clustering
- **Data:** Synthetic Unilever invoice data with realistic business features
- **Inspiration:** [Game Theory based Clustering](https://www.mit.edu/~vgarg/tkde-final.pdf)
""")
