import numpy as np
from scipy import sparse

def ensure_dense_array(X):
    """
    Ensure that the input is a dense numpy array.
    Handles sparse matrices, pandas DataFrames, and other array-like objects.
    """
    if X is None:
        return None
    
    # Handle sparse matrices
    if sparse.issparse(X):
        X = X.toarray()
    
    # Convert to numpy array and ensure float64 dtype
    X = np.asarray(X, dtype=np.float64)
    
    return X

def safe_len(X):
    """
    Safely get the length (number of rows) of an array or matrix.
    Works with both dense and sparse matrices.
    """
    if X is None:
        return 0
    
    if hasattr(X, 'shape'):
        return X.shape[0]
    
    # Fallback for other types
    try:
        return len(X)
    except TypeError:
        return 0

def validate_clustering_input(X):
    """
    Validate and prepare input data for clustering algorithms.
    Returns a dense numpy array ready for clustering.
    """
    if X is None:
        raise ValueError("Input data cannot be None")
    
    # Ensure dense array
    X = ensure_dense_array(X)
    
    # Check for valid shape
    if X.ndim != 2:
        raise ValueError(f"Input must be 2D array, got {X.ndim}D")
    
    if X.shape[0] == 0:
        raise ValueError("Input array is empty")
    
    if X.shape[1] == 0:
        raise ValueError("Input array has no features")
    
    # Check for NaN or infinite values
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values")
    
    return X 