import mlflow
import os
from models.ml_model import train_ml
from models.dl_model import train_dl
from models.qml_model import train_qml

def main():
    # Set the tracking URI to the Kubernetes service name
    # If running locally, you can change this to http://localhost:5000
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"Using MLflow Tracking URI: {tracking_uri}")
    
    print("\n--- Training Machine Learning Model ---")
    try:
        train_ml()
    except Exception as e:
        print(f"ML Training failed: {e}")

    print("\n--- Training Deep Learning Model ---")
    try:
        train_dl()
    except Exception as e:
        print(f"DL Training failed: {e}")

    print("\n--- Training Quantum ML Model ---")
    try:
        train_qml()
    except Exception as e:
        print(f"QML Training failed: {e}")

    print("\n✅ All training completed. Check MLflow UI at http://localhost:5000")

if __name__ == "__main__":
    main()
