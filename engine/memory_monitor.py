"""
Memory pressure monitor for MLX-Serve.

Watches KV cache utilization and triggers preemption when
memory pressure is high. Runs as a background asyncio task.

Pressure levels:
  GREEN  < 70% used  — normal operation
  YELLOW 70-85% used — warn, start being selective
  RED    > 85% used  — preempt lowest priority sequences
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from engine.kv_cache import KVCacheManager

logger = logging.getLogger(__name__)


class PressureLevel(Enum):
    GREEN  = "green"
    YELLOW = "yellow"
    RED    = "red"


@dataclass
class MemorySnapshot:
    timestamp: float
    total_blocks: int
    used_blocks: int
    free_blocks: int
    utilization: float
    pressure: PressureLevel
    active_sequences: int

    def to_dict(self) -> dict:
        return {
            "timestamp":        self.timestamp,
            "total_blocks":     self.total_blocks,
            "used_blocks":      self.used_blocks,
            "free_blocks":      self.free_blocks,
            "utilization_pct":  round(self.utilization * 100, 1),
            "pressure":         self.pressure.value,
            "active_sequences": self.active_sequences,
        }


class MemoryMonitor:
    def __init__(
        self,
        kv_mgr: KVCacheManager,
        poll_interval: float = 1.0,
        yellow_threshold: float = 0.70,
        red_threshold: float = 0.85,
    ):
        self.kv_mgr           = kv_mgr
        self.poll_interval    = poll_interval
        self.yellow_threshold = yellow_threshold
        self.red_threshold    = red_threshold

        self._running    = False
        self._task: Optional[asyncio.Task] = None
        self._latest: Optional[MemorySnapshot] = None
        self._history: list = []
        self._max_history = 60  # last 60 snapshots

        # Callbacks triggered on pressure change
        self._on_yellow: list[Callable] = []
        self._on_red: list[Callable] = []
        self._on_green: list[Callable] = []
        self._last_pressure = PressureLevel.GREEN

        # Stats
        self._yellow_events = 0
        self._red_events = 0
        self._preemptions = 0

    def on_yellow(self, fn: Callable):
        self._on_yellow.append(fn)

    def on_red(self, fn: Callable):
        self._on_red.append(fn)

    def on_green(self, fn: Callable):
        self._on_green.append(fn)

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Memory monitor started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        while self._running:
            try:
                snap = self._snapshot()
                self._latest = snap
                self._history.append(snap)
                if len(self._history) > self._max_history:
                    self._history.pop(0)

                # Fire callbacks on pressure level change
                if snap.pressure != self._last_pressure:
                    logger.info(
                        f"Memory pressure: {self._last_pressure.value} → "
                        f"{snap.pressure.value} "
                        f"({snap.utilization_pct:.1f}% used, "
                        f"{snap.free_blocks} blocks free)"
                        if hasattr(snap, 'utilization_pct')
                        else f"({snap.utilization*100:.1f}% used)"
                    )
                    if snap.pressure == PressureLevel.YELLOW:
                        self._yellow_events += 1
                        for fn in self._on_yellow:
                            asyncio.create_task(fn(snap))
                    elif snap.pressure == PressureLevel.RED:
                        self._red_events += 1
                        for fn in self._on_red:
                            asyncio.create_task(fn(snap))
                    elif snap.pressure == PressureLevel.GREEN:
                        for fn in self._on_green:
                            asyncio.create_task(fn(snap))
                    self._last_pressure = snap.pressure

            except Exception as e:
                logger.error(f"Memory monitor error: {e}")

            await asyncio.sleep(self.poll_interval)

    def _snapshot(self) -> MemorySnapshot:
        s = self.kv_mgr.stats()
        util = s["utilization"]
        if util >= self.red_threshold:
            pressure = PressureLevel.RED
        elif util >= self.yellow_threshold:
            pressure = PressureLevel.YELLOW
        else:
            pressure = PressureLevel.GREEN

        return MemorySnapshot(
            timestamp=time.time(),
            total_blocks=s["total_blocks"],
            used_blocks=s["used_blocks"],
            free_blocks=s["free_blocks"],
            utilization=util,
            pressure=pressure,
            active_sequences=s["active_sequences"],
        )

    def current(self) -> Optional[dict]:
        if self._latest is None:
            return None
        return self._latest.to_dict()

    def peak_utilization(self) -> float:
        if not self._history:
            return 0.0
        return max(s.utilization for s in self._history)

    def avg_utilization(self) -> float:
        if not self._history:
            return 0.0
        return round(sum(s.utilization for s in self._history) / len(self._history), 3)

    def stats(self) -> dict:
        return {
            "current":          self.current(),
            "peak_utilization": round(self.peak_utilization() * 100, 1),
            "avg_utilization":  round(self.avg_utilization() * 100, 1),
            "yellow_events":    self._yellow_events,
            "red_events":       self._red_events,
            "preemptions":      self._preemptions,
            "history_size":     len(self._history),
        }

    def record_preemption(self):
        self._preemptions += 1


# Global singleton — wired up in server startup
monitor: Optional[MemoryMonitor] = None
