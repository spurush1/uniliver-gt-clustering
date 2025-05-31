
# 🚀 Unilever Invoice Clustering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

This project compares **Game Theory-based Clustering** with traditional methods like KMeans, DBSCAN, and Agglomerative on realistic Unilever invoice data.

---

## 📦 Features

- Upload invoice CSVs with real-world business features
- Choose clustering method: `KMeans`, `DBSCAN`, `Agglomerative`, `Game Theory`
- Visualize clusters and get silhouette scores
- Download enriched output with cluster labels
- Backend built with **FastAPI**, frontend with **Streamlit**

---

## 🚀 Deployment (Streamlit Cloud)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin master
   ```

2. **Go to** [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“New app”**
   - Repo: `your-username/unilever-clustering-gt`
   - Branch: `master`
   - Main file: `streamlit_app/app.py`

---

## 📊 Management Report

📄 Available in `reports/management_summary.md`  
📥 [Download PDF Version](../management_summary.pdf)

---

## 📂 Folder Structure

```
unilever-clustering-gt/
├── app/               ← FastAPI backend
├── streamlit_app/     ← Streamlit UI
├── reports/           ← Summary report
├── app/data/          ← Synthetic invoice data
├── requirements.txt
├── README.md
```
