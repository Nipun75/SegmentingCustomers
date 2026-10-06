import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

def load_data(path):
    df = pd.read_csv(path)
    return df

def validate_data(df):
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    return df

def prepare_features(df):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler
