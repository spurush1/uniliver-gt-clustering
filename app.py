
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from services.preprocess import prepare_features, full_preprocess
from services.traditional import run_kmeans
from models.game_theory import GameTheoryClusterer

st.set_page_config(page_title="Unilever Invoice Clustering", layout="wide")

st.title("🧠 Unilever Invoice Clustering App")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

tab1, tab2 = st.tabs(["📊 Individual Clustering", "🔁 Comparison View"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X_full, df_encoded = full_preprocess(df)

    with tab1:
        st.write("### Preview of Uploaded Data", df.head())
        method = st.selectbox("Select Clustering Method", ["KMeans", "Game Theory"])

        if method == "KMeans":
            labels = run_kmeans(X_full, n_clusters=3)
        else:
            threshold = st.slider("Game Theory Threshold", 0.1, 1.0, 0.4)
            model = GameTheoryClusterer(X_full)
            labels = model.fit(threshold=threshold)

        df["cluster"] = labels
        st.write("### Clustered Data", df.head())

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_full)
        fig, ax = plt.subplots()
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2", ax=ax)
        ax.set_title("Cluster Plot (2D PCA Projection)")
        st.pyplot(fig)

        unique_labels = set(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(X_full):
            score = silhouette_score(X_full, labels)
            st.success(f"Silhouette Score: {score:.2f}")
        else:
            st.warning("Silhouette score could not be computed (only one cluster detected).")

        st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")

    with tab2:
        kmeans_labels = run_kmeans(X_full, n_clusters=3)
        gt_model = GameTheoryClusterer(X_full)
        gt_labels = gt_model.fit(threshold=0.4)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        reduced = PCA(n_components=2).fit_transform(X_full)

        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=kmeans_labels, ax=axes[0], palette="Set2").set(title="KMeans")
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=gt_labels, ax=axes[1], palette="Set1").set(title="Game Theory")

        st.pyplot(fig)

        st.markdown("### 📊 Cluster Label Breakdown by Method")
        st.write("KMeans Cluster Counts:", pd.Series(kmeans_labels).value_counts())
        st.write("Game Theory Cluster Counts:", pd.Series(gt_labels).value_counts())
