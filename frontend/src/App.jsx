import React, { useState } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const featureNames = [
  "Age", "Sex (1=M, 0=F)", "Chest Pain (0-3)", "Resting BP", "Cholesterol",
  "Fasting Sugar (>120)", "Resting ECG (0-2)", "Max Heart Rate", "Exercise Angina",
  "Oldpeak", "Slope (0-2)", "CA (0-4)", "Thal (0-3)"
];

function App() {
  const [features, setFeatures] = useState(Array(13).fill(0));
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = parseFloat(value);
    setFeatures(newFeatures);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/predict', { features });
      setResults(response.data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert("Failed to get prediction from backend. Make sure the FastAPI server is running.");
    }
    setLoading(false);
  };

  const chartData = results ? {
    labels: ['ML (Random Forest)', 'DL (PyTorch MLP)', 'QML (PennyLane VQC)'],
    datasets: [
      {
        label: 'Confidence Score',
        data: [results.ml.confidence, results.dl.confidence, results.qml.confidence],
        backgroundColor: [
          'rgba(59, 130, 246, 0.6)',
          'rgba(147, 51, 234, 0.6)',
          'rgba(236, 72, 153, 0.6)',
        ],
        borderColor: [
          'rgba(59, 130, 246, 1)',
          'rgba(147, 51, 234, 1)',
          'rgba(236, 72, 153, 1)',
        ],
        borderWidth: 1,
      },
    ],
  } : null;

  return (
    <div className="App">
      <header>
        <h1>QuantumMed AI</h1>
        <p style={{ color: '#94a3b8', marginBottom: '2rem' }}>
          Heart Disease Prediction Comparison: Classical vs Deep vs Quantum Paradigms
        </p>
      </header>

      <div className="grid-container">
        <div className="card">
          <h3>Patient Clinical Data</h3>
          <form onSubmit={handleSubmit} style={{ textAlign: 'left' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              {featureNames.map((name, i) => (
                <div key={i}>
                  <label style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>{name}</label>
                  <input
                    type="number"
                    step="any"
                    value={features[i]}
                    onChange={(e) => handleChange(i, e.target.value)}
                    required
                  />
                </div>
              ))}
            </div>
            <button type="submit" style={{ marginTop: '1.5rem', width: '100%' }} disabled={loading}>
              {loading ? "Analyzing..." : "Predict Health Status"}
            </button>
          </form>
        </div>

        <div className="card">
          <h3>Analysis Results</h3>
          {!results && <p style={{ color: '#64748b' }}>Enter patient data and submit to see predictions.</p>}
          
          {results && (
            <div>
              <div style={{ display: 'grid', gap: '1rem', marginBottom: '2rem' }}>
                <ResultItem 
                  title="Classical ML" 
                  pred={results.ml.prediction} 
                  conf={results.ml.confidence} 
                  color="#3b82f6" 
                />
                <ResultItem 
                  title="Deep Learning" 
                  pred={results.dl.prediction} 
                  conf={results.dl.confidence} 
                  color="#a855f7" 
                />
                <ResultItem 
                  title="Quantum ML" 
                  pred={results.qml.prediction} 
                  conf={results.qml.confidence} 
                  color="#ec4899" 
                />
              </div>
              
              <div style={{ height: '250px' }}>
                <Bar 
                  data={chartData} 
                  options={{ 
                    responsive: true, 
                    maintainAspectRatio: false,
                    scales: { y: { beginAtZero: true, max: 1 } },
                    plugins: { legend: { display: false } }
                  }} 
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultItem({ title, pred, conf, color }) {
  const isHealthy = pred === 0;
  return (
    <div className="result-card" style={{ borderLeft: `4px solid ${color}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: '600' }}>{title}</span>
        <span style={{ 
          padding: '2px 8px', 
          borderRadius: '4px', 
          fontSize: '0.75rem', 
          backgroundColor: isHealthy ? '#065f46' : '#991b1b',
          color: 'white'
        }}>
          {isHealthy ? "HEALTHY" : "AT RISK"}
        </span>
      </div>
      <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '4px' }}>
        Confidence: {(conf * 100).toFixed(1)}%
      </div>
      <div style={{ width: '100%', height: '4px', backgroundColor: '#334155', marginTop: '8px', borderRadius: '2px' }}>
        <div style={{ width: `${conf * 100}%`, height: '100%', backgroundColor: color, borderRadius: '2px' }}></div>
      </div>
    </div>
  );
}

export default App;
