import pennylane as qml
from pennylane import numpy as np
import pickle
import mlflow
import os
import sys

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessor import load_and_preprocess_data

# Use 4 qubits for 4 PCA features
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(weights, features):
    # Angle Encoding
    qml.AngleEmbedding(features, wires=range(n_qubits))
    # Variational layers
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, features):
    return circuit(weights, features) + bias

def cost(weights, bias, features, labels):
    predictions = [variational_classifier(weights, bias, f) for f in features]
    # Simple MSE-like loss for binary classification
    return np.mean((np.array(predictions) - labels)**2)

def train_qml():
    _, _, X_train_pca, X_test_pca, y_train, y_test, _, pca = load_and_preprocess_data()
    
    base_path = os.path.dirname(__file__)
    # Save PCA
    with open(os.path.join(base_path, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    
    # Preprocess labels to [-1, 1] for PauliZ output
    y_train_q = np.array(y_train.values * 2 - 1, requires_grad=False)
    y_test_q = np.array(y_test.values * 2 - 1, requires_grad=False)
    X_train_q = np.array(X_train_pca, requires_grad=False)
    X_test_q = np.array(X_test_pca, requires_grad=False)

    # Initialize weights
    n_layers = 2
    weights = np.random.randn(n_layers, n_qubits, requires_grad=True)
    bias = np.array(0.0, requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=0.1)
    batch_size = 5
    
    mlflow.set_experiment("QuantumMed_QML")
    with mlflow.start_run():
        epochs = 20 # Reduced for speed in demo
        for it in range(epochs):
            # Sample batch
            batch_index = np.random.randint(0, len(X_train_q), (batch_size,))
            X_batch = X_train_q[batch_index]
            y_batch = y_train_q[batch_index]
            
            weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, y_batch)
            
            if it % 5 == 0:
                c = cost(weights, bias, X_test_q, y_test_q)
                print(f"Epoch {it} | Cost: {c:f}")

        # Accuracy
        predictions = [np.sign(variational_classifier(weights, bias, f)) for f in X_test_q]
        acc = np.mean(predictions == y_test_q)
        
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("accuracy", acc)
        
        # Save model
        with open(os.path.join(base_path, "qml_model.pkl"), "wb") as f:
            pickle.dump({"weights": weights, "bias": bias}, f)
            
        print(f"QML VQC Trained. Accuracy: {acc:.4f}")

def predict_qml(features):
    base_path = os.path.dirname(__file__)
    # features: list of 13 floats
    with open(os.path.join(base_path, "qml_model.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_path, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    
    scaled = scaler.transform([features])
    pca_features = pca.transform(scaled)[0]
    
    val = variational_classifier(data["weights"], data["bias"], pca_features)
    # Map [-1, 1] back to [0, 1]
    prediction = 1 if val > 0 else 0
    # Confidence is absolute distance from boundary (scaled roughly)
    confidence = min(1.0, abs(float(val)))
    
    return int(prediction), float(confidence)

if __name__ == "__main__":
    train_qml()
