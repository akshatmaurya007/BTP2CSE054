"""
ml_classifier.py
----------------
Stage 2 of the Decision Engine.
Trains and runs a lightweight binary classifier to predict whether
offloading a task reduces total cost (latency + energy).

Model options:
  - Logistic Regression  (fast, interpretable)
  - MLP Neural Network   (more accurate, slightly heavier)

Input features:
  [data_size, cpu_cycles, max_delay, local_cpu_freq,
   edge_cpu_freq, uplink_rate, battery_level, network_latency]

Label:
  1 = offload is better
  0 = local is better

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "trained_classifier.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

FEATURE_NAMES = [
    "data_size",
    "cpu_cycles",
    "max_delay",
    "local_cpu_freq",
    "edge_cpu_freq",
    "uplink_rate",
    "battery_level",
    "network_latency",
]


class OffloadingClassifier:
    """
    Lightweight binary classifier for offloading decisions.
    Wraps sklearn models with a standard train/predict interface.
    """

    def __init__(self, model_type="logistic"):
        """
        model_type: 'logistic' (default) or 'mlp'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.is_trained = False

    def _build_model(self):
        if self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42
            )
        return LogisticRegression(max_iter=500, random_state=42)

    def train(self, X, y):
        """
        Train the classifier.
        X: np.ndarray of shape (n_samples, 8)
        y: np.ndarray of labels (0=local, 1=offload)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print(f"[Classifier] Trained {self.model_type} model on {len(y)} samples.")

    def evaluate(self, X_test, y_test):
        """Print classification report on test data."""
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n[Classifier] Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=["Local", "Offload"]))
        return acc

    def predict(self, features: list) -> int:
        """
        Predict for a single task.
        features: list of 8 values matching FEATURE_NAMES order.
        Returns: 0 (local) or 1 (offload)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() or load() first.")
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return int(self.model.predict(X_scaled)[0])

    def predict_proba(self, features: list) -> float:
        """Returns probability of offloading being better (class=1)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict_proba(X_scaled)[0][1])

    def save(self):
        """Save model and scaler to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[Classifier] Model saved to {MODEL_PATH}")

    def load(self):
        """Load model and scaler from disk."""
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        print(f"[Classifier] Model loaded from {MODEL_PATH}")
