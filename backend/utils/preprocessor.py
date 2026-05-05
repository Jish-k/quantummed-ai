import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def load_and_preprocess_data(filepath="data/heart.csv"):
    df = pd.read_csv(filepath)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA for QML (4 features)
    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return (X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, y_train, y_test, scaler, pca)

def get_preprocessors():
    # Helper to load saved scaler and pca
    try:
        import os
        base_path = os.path.join(os.path.dirname(__file__), "..", "models")
        with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(base_path, "pca.pkl"), "rb") as f:
            pca = pickle.load(f)
        return scaler, pca
    except FileNotFoundError:
        return None, None
