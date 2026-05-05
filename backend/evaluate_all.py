import os
import sys
import pandas as pd
import torch
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Add paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.preprocessor import load_and_preprocess_data
from models.ml_model import predict_ml
from models.dl_model import predict_dl
from models.qml_model import predict_qml

def evaluate():
    print("🚀 Loading data and pre-trained models for evaluation...")
    
    # Load test data
    _, X_test_scaled, _, _, _, y_test, _, _ = load_and_preprocess_data()
    
    print("\n" + "="*40)
    print("📊 MODEL ACCURACY SUMMARY")
    print("="*40)

    # 1. Evaluate Machine Learning (Random Forest)
    try:
        y_pred_ml = []
        for features in X_test_scaled:
            # We bypass the scaler in predict_ml for evaluation since data is already scaled
            with open("models/ml_model.pkl", "rb") as f:
                model = pickle.load(f)
            y_pred_ml.append(model.predict([features])[0])
        
        ml_acc = accuracy_score(y_test, y_pred_ml)
        print(f"✅ [ML]  Random Forest Accuracy : {ml_acc*100:.2f}%")
    except Exception as e:
        print(f"❌ [ML]  Evaluation failed: {e}")

    # 2. Evaluate Deep Learning (MLP)
    try:
        y_pred_dl = []
        # Load the raw X_test_scaled and convert to tensor
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        # Using the model's internal prediction logic
        from models.dl_model import HeartMLP
        model_dl = HeartMLP()
        model_dl.load_state_dict(torch.load("models/dl_model.pth"))
        model_dl.eval()
        with torch.no_grad():
            outputs = model_dl(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_dl = predicted.numpy()
            
        dl_acc = accuracy_score(y_test, y_pred_dl)
        print(f"✅ [DL]  Neural Network Accuracy: {dl_acc*100:.2f}%")
    except Exception as e:
        print(f"❌ [DL]  Evaluation failed: {e}")

    # 3. Evaluate Quantum ML (VQC)
    try:
        # Use a subset for QML evaluation as it can be slow
        subset_size = 20
        X_test_subset = X_test_scaled[:subset_size]
        y_test_subset = y_test[:subset_size]
        
        # We need to use the PCA features for QML
        _, _, _, X_test_pca, _, _, _, _ = load_and_preprocess_data()
        X_test_pca_subset = X_test_pca[:subset_size]
        
        from models.qml_model import variational_classifier
        with open("models/qml_model.pkl", "rb") as f:
            data_q = pickle.load(f)
        
        y_pred_qml = []
        for f in X_test_pca_subset:
            val = variational_classifier(data_q["weights"], data_q["bias"], f)
            y_pred_qml.append(1 if val > 0 else 0)
            
        q_acc = accuracy_score(y_test_subset, y_pred_qml)
        print(f"✅ [QML] Variational Quantum   : {q_acc*100:.2f}% (on subset)")
    except Exception as e:
        print(f"❌ [QML] Evaluation failed: {e}")

    print("="*40)
    print("Evaluation Complete!")

if __name__ == "__main__":
    evaluate()
