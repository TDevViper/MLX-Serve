from locust import HttpUser, task, between
import random

PROMPTS = [
    "What is machine learning?",
    "Explain neural networks in simple terms.",
    "What is the difference between CPU and GPU?",
    "How does attention mechanism work?",
    "What is gradient descent?",
    "Explain backpropagation.",
    "What is a transformer model?",
    "How does tokenization work in NLP?",
]

class MLXServeUser(HttpUser):
    wait_time = between(0.5, 2.0)
    host = "http://localhost:8000"

    @task(3)
    def chat_completion(self):
        prompt = random.choice(PROMPTS)
        self.client.post("/v1/chat/completions", json={
            "model": "mlx-community/Qwen1.5-0.5B-Chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 60,
        }, timeout=60)

    @task(1)
    def text_completion(self):
        self.client.post("/v1/completions", json={
            "model": "mlx-community/Qwen1.5-0.5B-Chat",
            "prompt": random.choice(PROMPTS),
            "max_tokens": 40,
        }, timeout=60)

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def metrics_check(self):
        self.client.get("/metrics")
