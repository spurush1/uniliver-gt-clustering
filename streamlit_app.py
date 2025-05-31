import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🎮 GT Clustering vs Traditional Methods",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Game Theory Clustering: Demonstrating Superiority")

st.write("Testing basic deployment...")

# Test data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("app/data/invoices_realistic.csv")
        st.success(f"✅ Loaded data from app/data/ - Shape: {df.shape}")
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/invoices_realistic.csv")
            st.success(f"✅ Loaded data from data/ - Shape: {df.shape}")
            return df
        except FileNotFoundError:
            st.error("❌ Sample data file not found in both locations")
            return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Test data loading
st.write("Testing data loading...")
df = load_data()

if df is not None:
    st.write("**Data Preview:**")
    st.dataframe(df.head())
    
    # Test imports
    st.write("Testing imports...")
    
    try:
        from app.services.preprocess import full_preprocess
        st.success("✅ Successfully imported preprocess")
        
        from app.models.game_theory import GameTheoryClusterer
        st.success("✅ Successfully imported GameTheoryClusterer")
        
        from app.services.traditional import run_kmeans
        st.success("✅ Successfully imported traditional methods")
        
        st.write("🎉 All imports successful!")
        
        # Test basic preprocessing
        try:
            X_full, df_encoded = full_preprocess(df)
            st.success(f"✅ Preprocessing successful - Shape: {X_full.shape}")
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ Import failed: {str(e)}")
        st.write("Import error details:", str(e))

else:
    st.write("❌ Cannot proceed without data")

st.write("Basic deployment test complete!") 