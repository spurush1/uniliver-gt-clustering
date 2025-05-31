
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    numeric_cols = ["quantity", "price_per_unit", "total_amount", "payment_days"]
    df = df[numeric_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled
