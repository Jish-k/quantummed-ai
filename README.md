# QuantumMed AI 🏥🤖⚛️

QuantumMed AI is a full-stack health prediction platform that compares three different AI paradigms: **Machine Learning (ML)**, **Deep Learning (DL)**, and **Quantum Machine Learning (QML)** for heart disease prediction.

## 🚀 Key Features
- **Triple Paradigm Prediction**: Compare Random Forest (ML) vs. PyTorch MLP (DL) vs. PennyLane VQC (QML).
- **Quantum Integration**: Utilizes a Variational Quantum Circuit (VQC) with PCA-reduced features.
- **Modern UI**: React-based dashboard with glassmorphism aesthetics and real-time Chart.js visualizations.
- **Experiment Tracking**: Integrated with MLflow for model performance monitoring.
- **Full Containerization**: Docker and Docker Compose support for seamless deployment.

## 🛠️ Tech Stack
- **Backend**: FastAPI, PyTorch, PennyLane, scikit-learn, MLflow.
- **Frontend**: React (Vite), Axios, Chart.js.
- **DevOps**: Docker, Docker Compose, Pytest.

## 🏃 How to Run

### Option 1: Docker (Recommended)
Run the entire stack (Backend, Frontend, and MLflow) with one command:
```bash
docker-compose up --build
```
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- MLflow Dashboard: `http://localhost:5001`

### Option 2: Local Development 

#### 1. Backend Setup 
```bash
cd backend
pip install -r requirements.txt
# Train models
python models/ml_model.py
python models/dl_model.py
python models/qml_model.py
# Start API
uvicorn main:app --reload 
```

#### 2. Frontend Setup 
```bash
cd frontend
npm install
npm run dev
```

## 📊 Dataset
Uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease), processing 13 clinical features for binary classification (Healthy vs. At Risk).

## 🧪 Testing 
```bash
python -m pytest backend/tests/
```
