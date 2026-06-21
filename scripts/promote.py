"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""

# TODO: Write a script that :
# 1- reads the F1 threshold from params.yaml,
# 2- fetches the weighted F1 score from the most recent MLflow evaluation run,
# 3- promotes the latest model version to a "production" alias if the score
# meets the threshold.
# 4- Print a clear message when the model does not qualify.
# Documentation: https://mlflow.org/docs/latest/model-registry.html

import os
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(
    client: MlflowClient, experiment_name: str
) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    # 1. Look up the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # 2. Search for the most recent run in this experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No evaluation runs found.")

    latest_run = runs[0]
    run_id = latest_run.info.run_id

    # 3. Get the f1_weighted metric
    f1_score = latest_run.data.metrics.get("f1_weighted", 0.0)

    return run_id, f1_score


def promote(client: MlflowClient, model_name: str):
    """
    Fetch all registered versions of the model, identify the latest one,
    and assign it the "production" alias.
    """
    # 1. Fetch all registered versions for the model
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(
            f"No registered versions found for model '{model_name}'"
        )

    # 2. Identify the latest version (highest version number)
    latest_version = max(int(v.version) for v in versions)

    # 3. Assign the "production" alias
    client.set_registered_model_alias(
        name=model_name, alias="production", version=str(latest_version)
    )
    print(
        f"Success! Model '{model_name}' version {latest_version} promoted to 'production'."
    )


def main():
    # Setup MLflow config
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "FinBERT_Sentiment_Analysis"
    )
    model_name = os.getenv("MODEL_NAME", "FinBERT_Model")

    client = MlflowClient(tracking_uri=tracking_uri)

    # 1- load the threshold from params.yaml
    params = load_params()
    # Note: Check your params.yaml file!
    # If the key is just "threshold" instead of "f1_threshold", change it below:
    threshold = params.get("f1_threshold", 0.6)

    # 2- call get_latest_f1()
    run_id, f1_score = get_latest_f1(client, experiment_name)
    print(f"Latest Run ID: {run_id}")
    print(f"F1 Score: {f1_score:.4f} (Required Threshold: {threshold})")

    # 3- call promote() only if the threshold is met
    if f1_score >= threshold:
        print("Threshold met! Promoting model...")
        promote(client, model_name)
    else:
        # 4- Print a clear message when the model does not qualify
        print(
            "Threshold NOT met. The model will NOT be promoted to production."
        )


if __name__ == "__main__":
    main()
