"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# 1. Add the parent directory so Python can find 'app' from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.utils import load_classifier

# Configure the Python logger using basicConfig
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log(level: str, message: str, **kwargs) -> None:
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "message": message,
        **kwargs,
    }
    logger.info(json.dumps(log_entry))


# Define Prometheus metrics
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["sentiment"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms", "Prediction latency in milliseconds",
    buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000]
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Total number of prediction errors"
)

load_dotenv()

classifiers = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        for text in texts:
            if len(text) > 2000:
                log(
                    "WARNING",
                    "Input text exceeds 2000 characters. It will be truncated.",
                    text_length=len(text),
                )

        start_time = time.time()
        predictions = classifiers["sentiment"](texts)
        latency_ms = (time.time() - start_time) * 1000
        PREDICTION_LATENCY.observe(latency_ms)

        results = []
        for text, pred in zip(texts, predictions):
            sentiment = pred["label"]
            confidence = pred["score"]

            PREDICTION_REQUESTS.labels(sentiment=sentiment).inc()
            log(
                "INFO",
                "Prediction successful",
                sentiment=sentiment,
                confidence=confidence,
                latency_ms=latency_ms,
            )

            results.append(
                PredictionResult(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence,
                    latency_ms=latency_ms,
                )
            )
        return results
    except Exception as e:
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction failed", error=str(e))
        raise e


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    if "sentiment" not in classifiers or classifiers["sentiment"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    if "sentiment" not in classifiers or classifiers["sentiment"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    results = run_predictions([request.text])
    return results[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    if "sentiment" not in classifiers or classifiers["sentiment"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(
            status_code=422, detail="Text list cannot be empty"
        )
    results = run_predictions(request.texts)
    return results


if __name__ == "__main__":
    import uvicorn

    # This block MUST run to start the server!
    uvicorn.run(app, host="0.0.0.0", port=8000)
