"""
Model loading utilities.
"""

import os
import json
import tempfile
from transformers import pipeline, AutoConfig
from huggingface_hub import hf_hub_download

def load_classifier():
    """
    Load the sentiment classifier.
    ...
    """
    model_source = os.getenv("MODEL_SOURCE", "mlflow")

    if model_source == "huggingface":
        hf_model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
        print(f"Loading model from HuggingFace: {hf_model_id}")

        # 1. Download raw config
        config_path = hf_hub_download(repo_id=hf_model_id, filename="config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # 2. Patch the validation error
        config_dict["id2label"] = {"0": "0", "1": "1", "2": "2"}
        config_dict["label2id"] = {"0": 0, "1": 1, "2": 2}

        # 3. Save the patched config to a temporary directory and load it
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_config_path = os.path.join(tmp_dir, "config.json")
            with open(tmp_config_path, "w") as f:
                json.dump(config_dict, f)
            
            hf_config = AutoConfig.from_pretrained(tmp_dir)

        return pipeline(
            "text-classification",
            model=hf_model_id,
            tokenizer=hf_model_id,
            config=hf_config,
            device="cpu"
        )
    
    # (Ensure no hidden spaces on any blank lines below this point!)
    import mlflow.transformers
    
    model_name = os.getenv("MODEL_NAME", "finbert")
    model_alias = os.getenv("MODEL_STAGE", "production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"Loading model from MLflow: models:/{model_name}@{model_alias}")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.transformers.load_model(
        f"models:/{model_name}@{model_alias}"
    )
