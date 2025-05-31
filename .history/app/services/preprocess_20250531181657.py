
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def prepare_features(df):
    numeric_cols = ["quantity", "price_per_unit", "total_amount", "payment_days"]
    scaler = StandardScaler()
    return scaler.fit_transform(df[numeric_cols])

def full_preprocess(df):
    import numpy as np
    from scipy import sparse
    
    def _ensure_dense_array(X):
        """Ensure that the input is a dense numpy array."""
        if X is None:
            return None
        if sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)
    
    def _validate_clustering_input(X):
        """Validate and prepare input data for clustering algorithms."""
        if X is None:
            raise ValueError("Input data cannot be None")
        X = _ensure_dense_array(X)
        if X.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {X.ndim}D")
        if X.shape[0] == 0:
            raise ValueError("Input array is empty")
        if X.shape[1] == 0:
            raise ValueError("Input array has no features")
        if not np.isfinite(X).all():
            raise ValueError("Input contains NaN or infinite values")
        return X
    
    numeric = ["quantity", "price_per_unit", "total_amount", "payment_days"]
    categorical = ["material", "uom", "vendor", "country_of_origin"]

    ct = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    pipeline = Pipeline([("prep", ct)])
    X = pipeline.fit_transform(df)

    # Ensure dense array and validate
    X = _ensure_dense_array(X)
    X = _validate_clustering_input(X)

    return X, pd.DataFrame(X)
