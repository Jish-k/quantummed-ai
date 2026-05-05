import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import os
import sys
import pickle

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessor import load_and_preprocess_data

class HeartMLP(nn.Module):
    def __init__(self, input_size=13):
        super(HeartMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        return self.network(x)

def train_dl():
    X_train, X_test, _, _, y_train, y_test, _, _ = load_and_preprocess_data()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train.values)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test.values)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    model = HeartMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    mlflow.set_experiment("QuantumMed_DL")
    with mlflow.start_run():
        epochs = 50
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_test_t).sum().item() / len(y_test_t)
            
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("accuracy", acc)
        
        base_path = os.path.dirname(__file__)
        torch.save(model.state_dict(), os.path.join(base_path, "dl_model.pth"))
        print(f"DL MLP Trained. Accuracy: {acc:.4f}")

def predict_dl(features):
    base_path = os.path.dirname(__file__)
    model = HeartMLP()
    model.load_state_dict(torch.load(os.path.join(base_path, "dl_model.pth")))
    model.eval()
    
    with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    scaled_features = scaler.transform([features])
    features_t = torch.FloatTensor(scaled_features)
    
    with torch.no_grad():
        outputs = model(features_t)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    return int(prediction.item()), float(confidence.item())

if __name__ == "__main__":
    train_dl()
