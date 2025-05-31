
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from app.services.preprocess import prepare_features
from app.services.traditional import run_kmeans, run_dbscan, run_agglomerative
from app.models.game_theory import GameTheoryClusterer

st.set_page_config(page_title="Unilever Invoice Clustering", layout="wide")

st.title("🧠 Unilever Invoice Clustering App")
st.write("Upload a CSV file with invoice data and compare clustering methods")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

method = st.selectbox("Select Clustering Method", ["KMeans", "DBSCAN", "Agglomerative", "Game Theory"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    X = prepare_features(df)

    if method == "KMeans":
        labels = run_kmeans(X)
    elif method == "DBSCAN":
        labels = run_dbscan(X)
    elif method == "Agglomerative":
        labels = run_agglomerative(X)
    else:
        model = GameTheoryClusterer(X)
        labels = model.fit()

    df["cluster"] = labels
    st.write("### Clustered Data", df.head())

    # Scatter plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="Set2", ax=ax)
    ax.set_title("Cluster Scatter Plot (first 2 features)")
    st.pyplot(fig)

    # Silhouette score
    try:
        score = silhouette_score(X, labels)
        st.success(f"Silhouette Score: {score:.2f}")
    except Exception as e:
        st.warning(f"Could not compute silhouette score: {str(e)}")

    # Download button
    st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")
