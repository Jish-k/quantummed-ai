# QuantumMed AI: Technical Model Details

This document provides a deep dive into the three AI paradigms implemented in the project.

## 🚀 Quick Reference
| Paradigm | Specific Model | Working Principle |
| :--- | :--- | :--- |
| **ML** | **Random Forest** | Voting ensemble of 100 Decision Trees |
| **DL** | **Multi-Layer Perceptron** | Feed-forward Neural Network with 3 layers |
| **QML** | **Variational Quantum Circuit** | Parameterized Quantum Circuit with Entanglement |

---

## 1. Machine Learning (ML) 🌲
**Algorithm**: Random Forest Classifier
**Library**: `scikit-learn`

### How it Works:
- **Architecture**: An ensemble of 100 Decision Trees.
- **Logic**: Each tree looks at different symptoms (features) and makes a choice. The final prediction is the "majority vote" of all 100 trees. This makes it extremely stable and accurate.
- **Preprocessing**: Uses `StandardScaler` to make sure all units (years, mmHg, mg/dl) are treated equally by the trees.

### Why use it?
Random Forest is the "gold standard" for tabular medical data. It is highly robust, handles outliers well, and provides a reliable baseline for comparing more complex models.

---

## 2. Deep Learning (DL) 🧠
**Algorithm**: Multi-Layer Perceptron (MLP) / Artificial Neural Network
**Library**: `PyTorch`

### How it Works:
- **Architecture**: A digital "Brain" with 3 layers.
- **Logic**: Data flows through layers of neurons. Each neuron has a "weight" that it adjusts during training. If a combination of high blood pressure and specific age is common in sick patients, the neurons "strengthen" that connection.
- **Optimization**: Uses the **Adam** optimizer to "learn" from its mistakes (loss) during 50 epochs of training.

### Why use it?
Neural networks can learn incredibly complex, non-linear relationships that simple algorithms might miss. It excels at finding patterns in how different clinical factors (like high BP combined with specific ECG results) interact.

---

## 3. Quantum Machine Learning (QML) ⚛️
**Algorithm**: Variational Quantum Circuit (VQC)
**Library**: `PennyLane`

### How it Works:
- **Architecture**: A 4-Qubit Variational Circuit.
- **Logic**: It turns patient data into "Quantum States". It uses **Quantum Entanglement** (the `BasicEntanglerLayers`) to let qubits influence each other in ways that normal computer bits cannot. This allows the model to explore a much larger "mathematical space" to find disease patterns.
- **Output**: It measures the state of the qubits at the end to get a final probability.

### Why use it?
QML is an experimental frontier. We use it here to demonstrate how "Quantum Advantage" might eventually allow us to find subtle correlations in medical data that classical bits (0 and 1) cannot represent.

---

## 📊 Summary Comparison

| Feature | Machine Learning | Deep Learning | Quantum ML |
| :--- | :--- | :--- | :--- |
| **Logic** | Decision Trees | Neurons & Weights | Qubits & Gates |
| **Strength** | Interpretability | Complexity | Probability & Entanglement |
| **Best For** | Fast baselines | Large datasets | Research & High-dim data |
