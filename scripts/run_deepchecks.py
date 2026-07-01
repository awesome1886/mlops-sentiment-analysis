"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import pandas as pd
import yaml
from app.utils import load_classifier
from dotenv import load_dotenv

# Import Deepchecks NLP modules
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PropertyDrift, PredictionDrift

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def main():
    params = load_params()
    property_drift_threshold = params["property_drift_threshold"]
    prediction_drift_threshold = params["prediction_drift_threshold"]

    print("Loading production model...")
    # Force HuggingFace model source for this check
    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    # 1. Create TextData objects
    print("Creating TextData objects and calculating properties...")
    test_dataset = TextData(test_texts, task_type="text_classification")
    stream_dataset = TextData(stream_texts, task_type="text_classification")

    # Calculate built-in NLP properties (like Sentiment, length, etc.)
    test_dataset.calculate_builtin_properties()
    stream_dataset.calculate_builtin_properties()

    # 2. Compute NLP Property Drift
    print("Computing NLP Property Drift...")
    prop_check = PropertyDrift()
    # test_dataset acts as our reference (train), stream_dataset is what we are checking (test)
    prop_result = prop_check.run(
        train_dataset=test_dataset, test_dataset=stream_dataset
    )

    # Extract the "Sentiment" drift score specifically
    sentiment_data = prop_result.value.get("Sentiment", 0.0)
    sentiment_drift = (
        sentiment_data.get("drift_score", 0.0)
        if isinstance(sentiment_data, dict)
        else float(sentiment_data)
    )

    print(
        f"Property Drift Score (Sentiment): {sentiment_drift:.4f} / Threshold: {property_drift_threshold}"
    )
    if sentiment_drift > property_drift_threshold:
        print("❌ ERROR: Property drift threshold exceeded!")
        sys.exit(1)

    # 3. Compute Prediction Drift
    print("Generating predictions for drift check...")
    test_preds = run_predictions(classifier, test_texts)
    stream_preds = run_predictions(classifier, stream_texts)

    print("Computing Prediction Drift...")
    pred_check = PredictionDrift()
    pred_result = pred_check.run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        train_predictions=test_preds,
        test_predictions=stream_preds,
    )

    # Extract the prediction drift score
    pred_drift_score = (
        pred_result.value
        if isinstance(pred_result.value, float)
        else pred_result.value.get("drift_score", 0.0)
    )

    print(
        f"Prediction Drift Score: {pred_drift_score:.4f} / Threshold: {prediction_drift_threshold}"
    )
    if pred_drift_score > prediction_drift_threshold:
        print("❌ ERROR: Prediction drift threshold exceeded!")
        sys.exit(1)

    print("✅ All drift checks passed successfully!")


if __name__ == "__main__":
    main()
