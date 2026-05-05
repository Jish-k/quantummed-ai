import unittest
import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ml_model import predict_ml
from models.dl_model import predict_dl
from models.qml_model import predict_qml

class TestModels(unittest.TestCase):
    def setUp(self):
        # Sample features (13)
        self.sample_features = [55, 1, 2, 130, 250, 0, 1, 150, 0, 2.3, 1, 0, 2]

    def test_ml_prediction(self):
        pred, conf = predict_ml(self.sample_features)
        self.assertIn(pred, [0, 1])
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_dl_prediction(self):
        pred, conf = predict_dl(self.sample_features)
        self.assertIn(pred, [0, 1])
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_qml_prediction(self):
        pred, conf = predict_qml(self.sample_features)
        self.assertIn(pred, [0, 1])
        self.assertTrue(0.0 <= conf <= 1.0)

if __name__ == '__main__':
    unittest.main()
