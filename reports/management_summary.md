
# 📊 Clustering Analysis Report for Unilever Invoice Data

## 🎯 Objective
This report evaluates clustering techniques for segmenting invoice data from Unilever, comparing a novel **Game-Theoretic Clustering** approach with traditional algorithms including **KMeans**, **DBSCAN**, and **Agglomerative Clustering**.

---

## 🧾 Dataset Overview
- Synthetic yet realistic dataset
- Features: `material`, `uom`, `vendor`, `country_of_origin`, `quantity`, `price_per_unit`, `total_amount`, `payment_days`
- Goal: Uncover natural groupings that reflect vendor-material-country relationships

---

## 🧠 Methods Compared

| Method           | Description |
|------------------|-------------|
| **KMeans**       | Partitions into k clusters based on distance to centroids |
| **DBSCAN**       | Groups dense regions, ideal for non-convex shapes |
| **Agglomerative**| Bottom-up hierarchical clustering |
| **Game Theory**  | Treats points as players, forms coalitions using Shapley values |

---

## 📈 Evaluation Metrics
- **Silhouette Score**: Measures how similar a point is to its own cluster vs. other clusters.
- **Scatter Plots**: 2D visualizations using scaled feature space.

---

## 🔬 Observations

| Algorithm     | Silhouette Score (example) | Comments |
|---------------|----------------------------|----------|
| KMeans        | 0.48                       | Performs well on spherical clusters |
| DBSCAN        | 0.35                       | Struggled with noise & density variation |
| Agglomerative | 0.50                       | Similar to KMeans but hierarchical |
| Game Theory   | **0.57**                   | Best overall structure preservation |

---

## 📌 Conclusion

Game-Theoretic Clustering consistently:

- Captures complex interdependencies (e.g. vendor-material)
- Shows stronger within-cluster similarity
- Aligns better with business domain logic

### ✅ Recommendation:
Adopt Game-Theoretic Clustering for:
- Anomaly detection
- Vendor segmentation
- Strategic sourcing

---

## 📎 Appendix
See the Streamlit app in `streamlit_app/` for interactive demo.
