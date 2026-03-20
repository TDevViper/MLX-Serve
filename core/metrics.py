import time
import threading
from dataclasses import dataclass, field
from typing import List
from collections import deque

@dataclass
class RequestMetric:
    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    elapsed: float
    tokens_per_sec: float
    timestamp: float = field(default_factory=time.time)
    status: str = "success"

class MetricsCollector:
    def __init__(self, window_size: int = 100):
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=window_size)
        self._total_requests = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._failed_requests = 0
        self._server_start = time.time()

    def record(self, metric: RequestMetric):
        with self._lock:
            self._history.append(metric)
            self._total_requests += 1
            self._total_tokens += metric.completion_tokens
            self._total_prompt_tokens += metric.prompt_tokens
            if metric.status == "failed":
                self._failed_requests += 1

    def current_throughput(self, window_secs: float = 60.0) -> float:
        """Tokens per second over the last window_secs."""
        now = time.time()
        with self._lock:
            recent = [m for m in self._history if now - m.timestamp <= window_secs]
        if not recent:
            return 0.0
        total_tokens = sum(m.completion_tokens for m in recent)
        return round(total_tokens / window_secs, 2)

    def avg_latency(self, window_secs: float = 60.0) -> float:
        now = time.time()
        with self._lock:
            recent = [m for m in self._history if now - m.timestamp <= window_secs]
        if not recent:
            return 0.0
        return round(sum(m.elapsed for m in recent) / len(recent), 3)

    def avg_tokens_per_sec(self, window_secs: float = 60.0) -> float:
        now = time.time()
        with self._lock:
            recent = [m for m in self._history if now - m.timestamp <= window_secs]
        if not recent:
            return 0.0
        return round(sum(m.tokens_per_sec for m in recent) / len(recent), 2)

    def p99_latency(self) -> float:
        with self._lock:
            if not self._history:
                return 0.0
            latencies = sorted(m.elapsed for m in self._history)
        idx = int(len(latencies) * 0.99)
        return round(latencies[min(idx, len(latencies)-1)], 3)

    def summary(self) -> dict:
        uptime = round(time.time() - self._server_start, 1)
        return {
            "uptime_seconds":       uptime,
            "total_requests":       self._total_requests,
            "failed_requests":      self._failed_requests,
            "total_tokens":         self._total_tokens,
            "total_prompt_tokens":  self._total_prompt_tokens,
            "throughput_tok_per_s": self.current_throughput(),
            "avg_latency_s":        self.avg_latency(),
            "avg_tokens_per_s":     self.avg_tokens_per_sec(),
            "p99_latency_s":        self.p99_latency(),
        }

    def prometheus(self, queue_stats: dict) -> str:
        s = self.summary()
        lines = [
            "# HELP mlx_serve_requests_total Total requests processed",
            "# TYPE mlx_serve_requests_total counter",
            f"mlx_serve_requests_total {s['total_requests']}",
            "",
            "# HELP mlx_serve_requests_failed Total failed requests",
            "# TYPE mlx_serve_requests_failed counter",
            f"mlx_serve_requests_failed {s['failed_requests']}",
            "",
            "# HELP mlx_serve_tokens_total Total tokens generated",
            "# TYPE mlx_serve_tokens_total counter",
            f"mlx_serve_tokens_total {s['total_tokens']}",
            "",
            "# HELP mlx_serve_throughput_tokens_per_second Current throughput",
            "# TYPE mlx_serve_throughput_tokens_per_second gauge",
            f"mlx_serve_throughput_tokens_per_second {s['throughput_tok_per_s']}",
            "",
            "# HELP mlx_serve_avg_latency_seconds Average request latency",
            "# TYPE mlx_serve_avg_latency_seconds gauge",
            f"mlx_serve_avg_latency_seconds {s['avg_latency_s']}",
            "",
            "# HELP mlx_serve_p99_latency_seconds P99 request latency",
            "# TYPE mlx_serve_p99_latency_seconds gauge",
            f"mlx_serve_p99_latency_seconds {s['p99_latency_s']}",
            "",
            "# HELP mlx_serve_queue_waiting Requests waiting in queue",
            "# TYPE mlx_serve_queue_waiting gauge",
            f"mlx_serve_queue_waiting {queue_stats.get('waiting', 0)}",
            "",
            "# HELP mlx_serve_queue_active Requests currently running",
            "# TYPE mlx_serve_queue_active gauge",
            f"mlx_serve_queue_active {queue_stats.get('active', 0)}",
            "",
            "# HELP mlx_serve_uptime_seconds Server uptime",
            "# TYPE mlx_serve_uptime_seconds counter",
            f"mlx_serve_uptime_seconds {s['uptime_seconds']}",
        ]
        return "\n".join(lines) + "\n"

    def prometheus_with_kv(self, queue_stats: dict, kv_stats: dict, mem_stats: dict) -> str:
        base = self.prometheus(queue_stats)
        kv_lines = [
            "",
            "# HELP mlx_serve_kv_blocks_total Total KV cache blocks",
            "# TYPE mlx_serve_kv_blocks_total gauge",
            f"mlx_serve_kv_blocks_total {kv_stats.get('total_blocks', 0)}",
            "",
            "# HELP mlx_serve_kv_blocks_used Used KV cache blocks",
            "# TYPE mlx_serve_kv_blocks_used gauge",
            f"mlx_serve_kv_blocks_used {kv_stats.get('used_blocks', 0)}",
            "",
            "# HELP mlx_serve_kv_utilization KV cache utilization 0-1",
            "# TYPE mlx_serve_kv_utilization gauge",
            f"mlx_serve_kv_utilization {kv_stats.get('utilization', 0)}",
            "",
            "# HELP mlx_serve_kv_evictions Total KV cache evictions",
            "# TYPE mlx_serve_kv_evictions counter",
            f"mlx_serve_kv_evictions {kv_stats.get('evictions', 0)}",
            "",
            "# HELP mlx_serve_memory_preemptions Total preemptions",
            "# TYPE mlx_serve_memory_preemptions counter",
            f"mlx_serve_memory_preemptions {mem_stats.get('preemptions', 0)}",
        ]
        return base + "\n".join(kv_lines) + "\n"

# Global singleton
metrics = MetricsCollector()
