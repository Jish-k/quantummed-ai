import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os
import sys

# Add parent dir to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessor import load_and_preprocess_data

def train_ml():
    X_train, X_test, _, _, y_train, y_test, scaler, _ = load_and_preprocess_data()
    
    base_path = os.path.dirname(__file__)
    # Save scaler
    with open(os.path.join(base_path, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    mlflow.set_experiment("QuantumMed_ML")
    with mlflow.start_run():
        n_estimators = 100
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        
        with open(os.path.join(base_path, "ml_model.pkl"), "wb") as f:
            pickle.dump(rf, f)
            
        print(f"ML Random Forest Trained. Accuracy: {acc:.4f}")

def predict_ml(features):
    base_path = os.path.dirname(__file__)
    # features: list of 13 floats
    with open(os.path.join(base_path, "ml_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    confidence = max(model.predict_proba(scaled_features)[0])
    return int(prediction), float(confidence)

if __name__ == "__main__":
    train_ml()
