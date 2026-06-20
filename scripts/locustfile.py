"""
Load test for the Sentiment Analysis API.

Run: locust -f locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure and start the test.
"""

from locust import HttpUser, task, between

class SentimentAPIUser(HttpUser):
    # Set a wait_time value
    wait_time = between(1, 3)

    @task(1)
    def health(self):
        self.client.get("/health")

    @task(3)
    def predict(self):
        self.client.post("/predict", json={"text": "The stock market is up today!"})

    @task(1)
    def predict_batch(self):
        self.client.post("/predict/batch", json={"texts": ["Good news", "Bad news"]})