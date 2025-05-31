
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def prepare_features(df):
    numeric_cols = ["quantity", "price_per_unit", "total_amount", "payment_days"]
    scaler = StandardScaler()
    return scaler.fit_transform(df[numeric_cols])

def full_preprocess(df):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils import ensure_dense_array, validate_clustering_input
    
    numeric = ["quantity", "price_per_unit", "total_amount", "payment_days"]
    categorical = ["material", "uom", "vendor", "country_of_origin"]

    ct = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    pipeline = Pipeline([("prep", ct)])
    X = pipeline.fit_transform(df)

    # Ensure dense array using our utility function
    X = ensure_dense_array(X)
    
    # Validate the input for clustering
    X = validate_clustering_input(X)

    return X, pd.DataFrame(X)
