"""Load test of POST /predict. Simulates machines sending readings.

    make loadtest        (API running on :8000, with a champion model and data)

Report the numbers you MEASURE (reports/loadtest_stats.csv) together with the machine they ran on.
"""

import random

from locust import HttpUser, between, task

MACHINES = [f"M{i:03d}" for i in range(1, 41)]


class Sensor(HttpUser):
    wait_time = between(0.05, 0.3)

    @task(6)
    def predict(self):
        self._predict(explain=False)

    @task(2)
    def predict_with_explanation(self):
        self._predict(explain=True)

    def _predict(self, explain):
        healthy = random.random() > 0.1
        self.client.post(
            "/predict",
            json={
                "machine_id": random.choice(MACHINES),
                "voltage": random.gauss(170 if healthy else 180, 8),
                "rotation": abs(random.gauss(446 if healthy else 400, 50)),
                "pressure": abs(random.gauss(100 if healthy else 112, 10)),
                "vibration": abs(random.gauss(40 if healthy else 52, 5)),
            },
            params={"explain": str(explain).lower()},
            name=f"/predict explain={str(explain).lower()}",
        )

    @task(1)
    def overview(self):
        self.client.get("/machines?limit=20", name="/machines")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
