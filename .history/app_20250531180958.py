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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from services.preprocess import prepare_features, full_preprocess
from services.traditional import run_kmeans, run_dbscan, run_agglomerative, auto_tune_dbscan, determine_optimal_clusters
from models.game_theory import GameTheoryClusterer

# Page configuration
st.set_page_config(
    page_title="🎮 GT Clustering vs Traditional Methods",
    page_icon="🎮",
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
    .gt-winner {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .traditional-method {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .superiority-badge {
        background: linear-gradient(90deg, #4caf50, #2e7d32);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎮 Game Theory Clustering: Demonstrating Superiority")

st.markdown("""
<div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
    <h3>🎯 Based on MIT Research: <a href="https://www.mit.edu/~vgarg/tkde-final.pdf" target="_blank">Game Theory based Clustering</a></h3>
    <p><strong>Game Theory Clustering</strong> uses <em>coalition formation</em> and <em>Shapley values</em> to create more natural, stable clusters compared to traditional methods.</p>
    <p><strong>Key Advantages:</strong></p>
    <ul>
        <li>📊 <strong>Higher Silhouette Scores:</strong> Better separated, more cohesive clusters</li>
        <li>🎯 <strong>Stable Coalitions:</strong> Data points naturally group based on mutual benefit</li>
        <li>⚖️ <strong>Fair Assignment:</strong> Shapley values ensure optimal cluster membership</li>
        <li>🔄 <strong>Adaptive:</strong> No need to pre-specify number of clusters</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("app/data/invoices_realistic.csv")
    except FileNotFoundError:
        st.error("Sample data file not found. Please check the file path.")
        return None

df = load_data()

if df is not None:
    # Sidebar configuration
    st.sidebar.header("⚙️ Clustering Configuration")
    
    # Show data info
    with st.sidebar.expander("📊 Dataset Info"):
        st.write(f"**Invoices:** {df.shape[0]}")
        st.write(f"**Features:** {df.shape[1]}")
        st.write(f"**Customers:** {df['customer_id'].nunique()}")

    # Preprocess data
    try:
        X_full, df_encoded = full_preprocess(df)
        st.sidebar.success(f"✅ Data preprocessed: {X_full.shape[1]} features")
        
        # Additional validation
        if X_full is None or X_full.shape[0] == 0:
            st.error("Preprocessed data is empty")
            st.stop()
            
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        st.error("Please check your data format and try again.")
        st.stop()

    # Parameters
    st.sidebar.subheader("🎮 Game Theory Parameters")
    gt_threshold = st.sidebar.slider("Coalition Threshold", 0.05, 0.8, 0.2, 0.05)
    gt_similarity = st.sidebar.selectbox("Similarity Metric", ["euclidean", "cosine"])
    gt_gamma = st.sidebar.slider("Gamma (Temperature)", 0.1, 5.0, 1.0, 0.1)
    
    st.sidebar.subheader("🔧 Traditional Methods")
    n_clusters_traditional = st.sidebar.slider("Number of Clusters", 2, 15, 5)
    auto_tune = st.sidebar.checkbox("Auto-tune Parameters", True)

    # Main content
    tab1, tab2, tab3 = st.tabs(["🏆 Superiority Demo", "🔬 Detailed Analysis", "📊 Cluster Insights"])

    with tab1:
        st.header("🏆 GT Clustering Superiority Demonstration")
        
        if st.button("🚀 Run Complete Comparison", type="primary", use_container_width=True):
            
            # Initialize results storage
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run Game Theory Clustering
            status_text.text("🎮 Running Game Theory clustering...")
            progress_bar.progress(20)
            
            try:
                gt_model = GameTheoryClusterer(X_full, gamma=gt_gamma, similarity_metric=gt_similarity)
                gt_labels = gt_model.fit(threshold=gt_threshold, max_coalition_size=8)
            except Exception as e:
                st.error(f"Error in Game Theory clustering: {str(e)}")
                st.error("Please try adjusting the parameters or check your data.")
                st.stop()
            
            results['Game Theory'] = {
                'labels': gt_labels,
                'n_clusters': len(np.unique(gt_labels)),
                'model': gt_model
            }
            
            # Run Traditional Methods
            methods = ['K-Means', 'DBSCAN', 'Agglomerative']
            
            for i, method in enumerate(methods):
                status_text.text(f"🔧 Running {method}...")
                progress_bar.progress(20 + (i + 1) * 20)
                
                if method == 'K-Means':
                    if auto_tune:
                        optimal_info = determine_optimal_clusters(X_full)
                        optimal_k = optimal_info['silhouette_method']
                        labels = run_kmeans(X_full, n_clusters=optimal_k)
                        n_clusters = optimal_k
                    else:
                        labels = run_kmeans(X_full, n_clusters=n_clusters_traditional)
                        n_clusters = n_clusters_traditional
                        
                elif method == 'DBSCAN':
                    if auto_tune:
                        labels, params = auto_tune_dbscan(X_full)
                        n_clusters = len(np.unique(labels[labels != -1]))
                    else:
                        labels = run_dbscan(X_full)
                        n_clusters = len(np.unique(labels[labels != -1]))
                        
                elif method == 'Agglomerative':
                    if auto_tune:
                        optimal_info = determine_optimal_clusters(X_full)
                        optimal_k = optimal_info['silhouette_method']
                        labels = run_agglomerative(X_full, n_clusters=optimal_k)
                        n_clusters = optimal_k
                    else:
                        labels = run_agglomerative(X_full, n_clusters=n_clusters_traditional)
                        n_clusters = n_clusters_traditional
                
                results[method] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'model': None
                }
            
            # Compute metrics
            status_text.text("📊 Computing performance metrics...")
            progress_bar.progress(90)
            
            for method_name, result in results.items():
                labels = result['labels']
                
                # Handle noise points in DBSCAN
                if method_name == 'DBSCAN':
                    mask = labels != -1
                    if np.sum(mask) > 10 and len(np.unique(labels[mask])) > 1:
                        silhouette = silhouette_score(X_full[mask], labels[mask])
                        calinski = calinski_harabasz_score(X_full[mask], labels[mask])
                    else:
                        silhouette = -1
                        calinski = 0
                else:
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(X_full, labels)
                        calinski = calinski_harabasz_score(X_full, labels)
                    else:
                        silhouette = -1
                        calinski = 0
                
                result['silhouette'] = silhouette
                result['calinski'] = calinski
                result['noise_points'] = np.sum(labels == -1) if method_name == 'DBSCAN' else 0
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display Results
            st.subheader("🏆 Performance Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for method, result in results.items():
                comparison_data.append({
                    'Method': method,
                    'Silhouette Score': result['silhouette'],
                    'Calinski-Harabasz': result['calinski'],
                    'Number of Clusters': result['n_clusters'],
                    'Noise Points': result['noise_points']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Find the best method for each metric
            best_silhouette = comparison_df.loc[comparison_df['Silhouette Score'].idxmax(), 'Method']
            best_calinski = comparison_df.loc[comparison_df['Calinski-Harabasz'].idxmax(), 'Method']
            
            # Display metrics with highlighting
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Silhouette Score (Higher = Better)")
                for _, row in comparison_df.iterrows():
                    method = row['Method']
                    score = row['Silhouette Score']
                    
                    if method == best_silhouette and method == 'Game Theory':
                        st.markdown(f"""
                        <div class="gt-winner">
                            🏆 <strong>{method}:</strong> {score:.4f} <span class="superiority-badge">WINNER</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif method == best_silhouette:
                        st.markdown(f"""
                        <div class="traditional-method">
                            🥇 <strong>{method}:</strong> {score:.4f}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write(f"**{method}:** {score:.4f}")
            
            with col2:
                st.markdown("### 🎯 Calinski-Harabasz Index (Higher = Better)")
                for _, row in comparison_df.iterrows():
                    method = row['Method']
                    score = row['Calinski-Harabasz']
                    
                    if method == best_calinski and method == 'Game Theory':
                        st.markdown(f"""
                        <div class="gt-winner">
                            🏆 <strong>{method}:</strong> {score:.2f} <span class="superiority-badge">WINNER</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif method == best_calinski:
                        st.markdown(f"""
                        <div class="traditional-method">
                            🥇 <strong>{method}:</strong> {score:.2f}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write(f"**{method}:** {score:.2f}")
            
            # Performance summary
            gt_silhouette = results['Game Theory']['silhouette']
            traditional_silhouettes = [results[method]['silhouette'] for method in methods if results[method]['silhouette'] > -1]
            
            if traditional_silhouettes:
                best_traditional_silhouette = max(traditional_silhouettes)
                improvement = ((gt_silhouette - best_traditional_silhouette) / best_traditional_silhouette) * 100
                
                if improvement > 0:
                    st.markdown(f"""
                    <div class="gt-winner">
                        <h3>🎉 Game Theory Clustering Superiority Confirmed!</h3>
                        <p><strong>Silhouette Score Improvement: +{improvement:.1f}%</strong> over best traditional method</p>
                        <p>GT clustering creates more cohesive and well-separated clusters through coalition formation!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"Traditional methods performed better in this case. GT improvement: {improvement:.1f}%")
            
            # Visualizations
            st.subheader("📈 Cluster Visualizations")
            
            # PCA for visualization
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_full)
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Game Theory Clustering', 'K-Means', 'DBSCAN', 'Agglomerative'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            colors = px.colors.qualitative.Set1
            
            methods_viz = ['Game Theory', 'K-Means', 'DBSCAN', 'Agglomerative']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for i, (method, pos) in enumerate(zip(methods_viz, positions)):
                labels = results[method]['labels']
                unique_labels = np.unique(labels)
                
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    if label == -1:  # Noise points
                        color = 'black'
                        name = 'Noise'
                    else:
                        color = colors[j % len(colors)]
                        name = f'Cluster {label}'
                    
                    fig.add_trace(
                        go.Scatter(
                            x=X_pca[mask, 0],
                            y=X_pca[mask, 1],
                            mode='markers',
                            marker=dict(color=color, size=4),
                            name=name,
                            showlegend=(i == 0),  # Only show legend for first subplot
                            legendgroup=name
                        ),
                        row=pos[0], col=pos[1]
                    )
            
            fig.update_layout(height=800, title="Clustering Results Comparison (PCA Projection)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance bar chart
            st.subheader("📊 Performance Metrics Comparison")
            
            metrics_fig = go.Figure()
            
            methods_list = list(comparison_df['Method'])
            silhouette_scores = list(comparison_df['Silhouette Score'])
            
            colors_bar = ['#4CAF50' if method == 'Game Theory' else '#FF9800' for method in methods_list]
            
            metrics_fig.add_trace(go.Bar(
                x=methods_list,
                y=silhouette_scores,
                marker_color=colors_bar,
                text=[f'{score:.4f}' for score in silhouette_scores],
                textposition='auto',
            ))
            
            metrics_fig.update_layout(
                title="Silhouette Score Comparison",
                xaxis_title="Clustering Method",
                yaxis_title="Silhouette Score",
                showlegend=False
            )
            
            st.plotly_chart(metrics_fig, use_container_width=True)

    with tab2:
        st.header("🔬 Detailed Individual Analysis")
        
        method = st.selectbox(
            "Select Clustering Method",
            ["Game Theory", "K-Means", "DBSCAN", "Agglomerative"],
            key="detailed_method"
        )
        
        if st.button("🔍 Analyze Method", key="analyze_individual"):
            with st.spinner(f"Analyzing {method}..."):
                
                if method == "Game Theory":
                    model = GameTheoryClusterer(X_full, gamma=gt_gamma, similarity_metric=gt_similarity)
                    labels = model.fit(threshold=gt_threshold, max_coalition_size=8)
                    
                    # Show GT-specific insights
                    st.subheader("🎮 Game Theory Insights")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Coalitions Formed", len(np.unique(labels)))
                    with col2:
                        st.metric("Average Coalition Size", f"{len(labels) / len(np.unique(labels)):.1f}")
                    with col3:
                        st.metric("Threshold Used", gt_threshold)
                    
                    # Show cluster statistics if available
                    if hasattr(model, 'get_cluster_statistics'):
                        try:
                            cluster_stats = model.get_cluster_statistics()
                            st.write("**Cluster Statistics:**")
                            for cluster_id, stats in cluster_stats.items():
                                st.write(f"Cluster {cluster_id}: {stats}")
                        except:
                            pass
                            
                elif method == "K-Means":
                    if auto_tune:
                        optimal_info = determine_optimal_clusters(X_full)
                        optimal_k = optimal_info['silhouette_method']
                        st.info(f"Auto-selected {optimal_k} clusters")
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
                        st.info(f"Auto-selected {optimal_k} clusters")
                        labels = run_agglomerative(X_full, n_clusters=optimal_k)
                    else:
                        labels = run_agglomerative(X_full, n_clusters=n_clusters_traditional)
                
                # Compute and display metrics
                if len(np.unique(labels)) > 1:
                    if method == 'DBSCAN' and -1 in labels:
                        mask = labels != -1
                        if np.sum(mask) > 10:
                            silhouette = silhouette_score(X_full[mask], labels[mask])
                            calinski = calinski_harabasz_score(X_full[mask], labels[mask])
                        else:
                            silhouette = -1
                            calinski = 0
                    else:
                        silhouette = silhouette_score(X_full, labels)
                        calinski = calinski_harabasz_score(X_full, labels)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Silhouette Score", f"{silhouette:.4f}")
                    with col2:
                        st.metric("Calinski-Harabasz", f"{calinski:.2f}")
                    with col3:
                        noise_points = np.sum(labels == -1) if method == 'DBSCAN' else 0
                        st.metric("Noise Points", noise_points)
                
                # Visualization
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_full)
                
                fig = px.scatter(
                    x=X_pca[:, 0], 
                    y=X_pca[:, 1], 
                    color=labels.astype(str),
                    title=f"{method} Clustering Results",
                    labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("📊 Business Insights from Clustering")
        
        if st.button("🔍 Generate Insights", key="generate_insights"):
            # Run GT clustering for insights
            model = GameTheoryClusterer(X_full, gamma=gt_gamma, similarity_metric=gt_similarity)
            labels = model.fit(threshold=gt_threshold, max_coalition_size=8)
            
            # Add cluster labels to original dataframe
            df_clustered = df.copy()
            df_clustered['GT_Cluster'] = labels
            
            st.subheader("🎯 Game Theory Cluster Characteristics")
            
            # Analyze each cluster
            for cluster_id in sorted(np.unique(labels)):
                cluster_data = df_clustered[df_clustered['GT_Cluster'] == cluster_id]
                
                with st.expander(f"📦 Coalition {cluster_id} ({len(cluster_data)} invoices)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Financial Summary:**")
                        st.write(f"- Total Amount: ${cluster_data['total_amount'].sum():,.2f}")
                        st.write(f"- Average Amount: ${cluster_data['total_amount'].mean():,.2f}")
                        st.write(f"- Average Payment Days: {cluster_data['payment_days'].mean():.1f}")
                        
                        st.write("**Top Materials:**")
                        top_materials = cluster_data['material'].value_counts().head(3)
                        for material, count in top_materials.items():
                            st.write(f"- {material}: {count} invoices")
                    
                    with col2:
                        st.write("**Geographic Distribution:**")
                        top_countries = cluster_data['country_of_origin'].value_counts().head(3)
                        for country, count in top_countries.items():
                            st.write(f"- {country}: {count} invoices")
                            
                        st.write("**Top Vendors:**")
                        top_vendors = cluster_data['vendor'].value_counts().head(3)
                        for vendor, count in top_vendors.items():
                            st.write(f"- {vendor}: {count} invoices")
            
            # Download option
            st.subheader("💾 Download Results")
            csv = df_clustered.to_csv(index=False)
            st.download_button(
                label="📥 Download Clustered Data",
                data=csv,
                file_name="gt_clustered_invoices.csv",
                mime="text/csv"
            )

else:
    st.error("Unable to load data. Please check the file path.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎮 <strong>Game Theory Clustering</strong> | Based on <a href="https://www.mit.edu/~vgarg/tkde-final.pdf" target="_blank">MIT Research</a></p>
    <p>Demonstrating superior clustering through coalition formation and Shapley values</p>
</div>
""", unsafe_allow_html=True)
