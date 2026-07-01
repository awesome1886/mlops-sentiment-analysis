"""
Production monitoring stream for FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint,
and logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root:
    python monitoring/stream.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', 8000)}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100    # delay between requests to simulate real traffic (ms)


def predict(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

# TODO (Task 6)
# Implement log_window() to aggregate results from one observation window and
# log them as MLflow metrics. For each window, compute and log:
# - Sentiment distribution (% positive, % negative, % neutral)
# - Average confidence score
# - Average latency in milliseconds
# Documentation: https://mlflow.org/docs/latest/tracking.html#logging-metrics
def log_window(window: list[dict], window_idx: int) -> None:
    """Log aggregated metrics for one observation window to MLflow."""
    if not window:
        return
        
    total = len(window)
    pos = sum(1 for w in window if w.get("sentiment") == "positive")
    neg = sum(1 for w in window if w.get("sentiment") == "negative")
    neu = sum(1 for w in window if w.get("sentiment") == "neutral")
    
    avg_conf = sum(w.get("confidence", 0.0) for w in window) / total
    avg_lat = sum(w.get("latency_ms", 0.0) for w in window) / total

    metrics = {
        "pct_positive": pos / total,
        "pct_negative": neg / total,
        "pct_neutral": neu / total,
        "avg_confidence": avg_conf,
        "avg_latency_ms": avg_lat
    }
    
    mlflow.log_metrics(metrics, step=window_idx)
# Implement main() to:
# 1. Load headlines from data/stream.csv
# 2. Start an MLflow run for monitoring
# 3. Loop through headlines, call predict() for each, and collect results in windows
# 4. Call log_window() every WINDOW_SIZE predictions to log metrics to MLflow
def main():
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # 1. Load headlines from data/stream.csv
    df = pd.read_csv("data/stream.csv")
    
    # 2. Start an MLflow run for monitoring
    with mlflow.start_run():
        window = []
        window_idx = 0
        
        # 3. Loop through headlines
        for text in df['headline']:
            time.sleep(SLEEP_MS / 1000.0)  # simulate real traffic delay
            
            try:
                result = predict(text)
                window.append(result)
            except Exception as e:
                print(f"API request failed: {e}")
                
            # 4. Call log_window() every WINDOW_SIZE predictions
            if len(window) == WINDOW_SIZE:
                log_window(window, window_idx)
                window = []  # Reset the window
                window_idx += 1

if __name__ == "__main__":
    main()
