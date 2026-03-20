import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ModelStats:
    model: str
    total_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_latency: float = 0.0
    failed: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_latency / self.total_requests, 3)

    def avg_tokens_per_req(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_tokens / self.total_requests, 1)

    def to_dict(self) -> dict:
        return {
            "model":               self.model,
            "total_requests":      self.total_requests,
            "total_tokens":        self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "failed_requests":     self.failed,
            "avg_latency_s":       self.avg_latency(),
            "avg_tokens_per_req":  self.avg_tokens_per_req(),
            "first_seen":          self.first_seen,
            "last_seen":           self.last_seen,
        }

class StatsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._models: Dict[str, ModelStats] = defaultdict(
            lambda: ModelStats(model="unknown")
        )

    def record(self, model: str, prompt_tokens: int,
               completion_tokens: int, latency: float, failed: bool = False):
        with self._lock:
            if model not in self._models:
                self._models[model] = ModelStats(model=model)
            s = self._models[model]
            s.total_requests      += 1
            s.total_tokens        += completion_tokens
            s.total_prompt_tokens += prompt_tokens
            s.total_latency       += latency
            s.last_seen            = time.time()
            if failed:
                s.failed += 1

    def all(self) -> list:
        with self._lock:
            return [s.to_dict() for s in self._models.values()]

    def for_model(self, model: str) -> dict:
        with self._lock:
            if model not in self._models:
                return {}
            return self._models[model].to_dict()

# Global singleton
stats = StatsTracker()
