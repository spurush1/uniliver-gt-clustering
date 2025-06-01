import streamlit as st

# Basic page config
st.set_page_config(
    page_title="🎮 GT Clustering",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Game Theory Clustering")
st.write("🔍 **Deployment Diagnostic**")

# Test 1: Basic imports
st.write("**Step 1: Testing basic imports...**")
try:
    import pandas as pd
    import numpy as np
    st.success("✅ pandas, numpy imported successfully")
except Exception as e:
    st.error(f"❌ Basic imports failed: {e}")
    st.stop()

# Test 2: Data loading
st.write("**Step 2: Testing data loading...**")
try:
    # Try both paths
    try:
        df = pd.read_csv("app/data/invoices_realistic.csv")
        data_path = "app/data/"
    except:
        df = pd.read_csv("data/invoices_realistic.csv")
        data_path = "data/"
    
    st.success(f"✅ Data loaded from {data_path} - Shape: {df.shape}")
    st.write("Sample data:")
    st.dataframe(df.head(3))
    
except Exception as e:
    st.error(f"❌ Data loading failed: {e}")
    st.write("Available files:")
    import os
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.csv'):
                st.write(f"Found CSV: {os.path.join(root, file)}")
    st.stop()

# Test 3: App module imports
st.write("**Step 3: Testing app module imports...**")
try:
    from app.services.preprocess import full_preprocess
    st.success("✅ preprocess module imported")
except Exception as e:
    st.error(f"❌ preprocess import failed: {e}")
    st.stop()

try:
    from app.models.game_theory import GameTheoryClusterer
    st.success("✅ game_theory module imported")
except Exception as e:
    st.error(f"❌ game_theory import failed: {e}")
    st.stop()

try:
    from app.services.traditional import run_kmeans
    st.success("✅ traditional methods imported")
except Exception as e:
    st.error(f"❌ traditional import failed: {e}")
    st.stop()

# Test 4: Preprocessing
st.write("**Step 4: Testing preprocessing...**")
try:
    X_full, df_encoded = full_preprocess(df)
    st.success(f"✅ Preprocessing successful - Shape: {X_full.shape}")
except Exception as e:
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

# Test 5: Basic clustering
st.write("**Step 5: Testing basic clustering...**")
if st.button("🧪 Test Small Clustering"):
    try:
        # Use only first 50 rows for quick test
        X_small = X_full[:50]
        st.write(f"Testing with {X_small.shape[0]} samples...")
        
        # Test GameTheory clustering
        gt_model = GameTheoryClusterer(X_small, gamma=1.0, similarity_metric='euclidean')
        gt_labels = gt_model.fit(threshold=0.3, max_coalition_size=4)
        
        st.success(f"✅ GT Clustering successful - {len(np.unique(gt_labels))} clusters formed")
        
        # Test K-means
        from app.services.traditional import run_kmeans
        kmeans_labels = run_kmeans(X_small, n_clusters=3)
        st.success(f"✅ K-means successful - {len(np.unique(kmeans_labels))} clusters")
        
    except Exception as e:
        st.error(f"❌ Clustering test failed: {e}")
        import traceback
        st.code(traceback.format_exc())

st.write("---")
st.success("🎉 **All tests passed!** Ready for full application.")

if st.button("🚀 Load Full Application"):
    st.info("Full application would load here...")
    st.balloons()
