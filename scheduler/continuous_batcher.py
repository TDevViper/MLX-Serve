"""
Continuous Batching Scheduler — fixed with prefix caching.
Each sequence now generates coherently using stream_with_cache.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SeqStatus(Enum):
    WAITING   = "waiting"
    RUNNING   = "running"
    DONE      = "done"
    PREEMPTED = "preempted"


@dataclass
class Sequence:
    seq_id:        str
    prompt:        str
    prompt_tokens: int
    max_tokens:    int
    temperature:   float
    stream:        bool
    created_at:    float = field(default_factory=time.time)
    status:        SeqStatus = SeqStatus.WAITING

    generated_tokens: int = 0
    output_text:      str = ""
    finish_reason:    str = ""
    token_queue:      Optional[asyncio.Queue] = field(default=None, repr=False)

    start_time:  Optional[float] = None
    finish_time: Optional[float] = None

    # Internal generator for token-by-token generation
    _gen: Optional[object] = field(default=None, repr=False)

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return round((self.finish_time or time.time()) - self.start_time, 3)

    def tokens_per_sec(self) -> float:
        e = self.elapsed()
        return round(self.generated_tokens / e, 1) if e > 0 else 0.0

    def is_finished(self) -> bool:
        return self.generated_tokens >= self.max_tokens

    def __post_init__(self):
        if self.stream:
            self.token_queue = asyncio.Queue()


@dataclass
class BatchStats:
    iteration:        int
    batch_size:       int
    waiting:          int
    tokens_generated: int
    elapsed_ms:       float
    timestamp:        float = field(default_factory=time.time)


class ContinuousBatcher:
    def __init__(
        self,
        max_batch_size: int = 4,
        max_tokens_per_batch: int = 2048,
        scheduler_interval: float = 0.0,
    ):
        self.max_batch_size       = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.scheduler_interval   = scheduler_interval

        self._waiting: asyncio.Queue       = asyncio.Queue()
        self._running: Dict[str, Sequence] = {}
        self._done:    Dict[str, Sequence] = {}

        self._running_task: Optional[asyncio.Task] = None
        self._active = False

        self._total_iterations       = 0
        self._total_tokens_generated = 0
        self._batch_stats: List[BatchStats] = []
        self._max_stats  = 100
        self._start_time = time.time()

    async def start(self):
        self._active = True
        self._running_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            f"Continuous batcher started — "
            f"max_batch={self.max_batch_size}"
        )

    async def stop(self):
        self._active = False
        if self._running_task:
            self._running_task.cancel()

    async def submit(self, seq: Sequence):
        await self._waiting.put(seq)

    async def wait_for_result(
        self, seq_id: str, timeout: float = 120.0
    ) -> Optional[Sequence]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if seq_id in self._done:
                return self._done[seq_id]
            await asyncio.sleep(0.05)
        return None

    async def _scheduler_loop(self):
        from engine.model_runner import runner
        while self._active:
            await self._admit_sequences(runner)
            if not self._running:
                await asyncio.sleep(0.01)
                continue
            t0 = time.time()
            tokens_this_iter = await self._run_one_step()
            elapsed_ms = (time.time() - t0) * 1000
            self._total_iterations       += 1
            self._total_tokens_generated += tokens_this_iter
            stat = BatchStats(
                iteration=self._total_iterations,
                batch_size=len(self._running),
                waiting=self._waiting.qsize(),
                tokens_generated=tokens_this_iter,
                elapsed_ms=round(elapsed_ms, 2),
            )
            self._batch_stats.append(stat)
            if len(self._batch_stats) > self._max_stats:
                self._batch_stats.pop(0)
            if self._total_iterations % 20 == 0:
                logger.info(
                    f"Batcher iter {self._total_iterations}: "
                    f"batch={len(self._running)}, "
                    f"waiting={self._waiting.qsize()}, "
                    f"tok/iter={tokens_this_iter}, "
                    f"{elapsed_ms:.1f}ms"
                )
            if self.scheduler_interval > 0:
                await asyncio.sleep(self.scheduler_interval)

    async def _admit_sequences(self, runner):
        """Admit waiting sequences and initialise their token generators."""
        while (
            len(self._running) < self.max_batch_size
            and not self._waiting.empty()
        ):
            try:
                seq = self._waiting.get_nowait()
                seq.status     = SeqStatus.RUNNING
                seq.start_time = time.time()
                # Create a persistent generator for this sequence
                seq._gen = self._make_generator(seq, runner)
                self._running[seq.seq_id] = seq
                logger.debug(f"Admitted [{seq.seq_id[:8]}]")
            except asyncio.QueueEmpty:
                break

    def _make_generator(self, seq: Sequence, runner):
        """
        Return a generator that yields one token at a time
        using stream_with_cache — coherent and efficient.
        """
        return runner.stream_with_cache(
            prompt=seq.prompt,
            max_tokens=seq.max_tokens,
            temperature=seq.temperature,
        )

    async def _run_one_step(self) -> int:
        """Advance every running sequence by one token."""
        loop = asyncio.get_event_loop()
        finished = []
        tokens_generated = 0

        for seq_id, seq in list(self._running.items()):
            token, is_done = await loop.run_in_executor(
                None, self._next_token, seq
            )
            if not is_done and token:
                seq.output_text      += token
                seq.generated_tokens += 1
                tokens_generated     += 1
                if seq.stream and seq.token_queue:
                    await seq.token_queue.put(token)

            if is_done or seq.is_finished():
                seq.status      = SeqStatus.DONE
                seq.finish_time = time.time()
                if seq.stream and seq.token_queue:
                    await seq.token_queue.put(None)
                finished.append(seq_id)
                logger.debug(
                    f"Finished [{seq_id[:8]}] — "
                    f"{seq.generated_tokens} tokens, "
                    f"{seq.tokens_per_sec()} tok/s"
                )

        for seq_id in finished:
            seq = self._running.pop(seq_id)
            self._done[seq_id] = seq

        return tokens_generated

    def _next_token(self, seq: Sequence):
        """Get next token from the sequence generator. Runs in thread."""
        try:
            return next(seq._gen)
        except StopIteration:
            return "", True
        except Exception as e:
            logger.error(f"Generator error [{seq.seq_id[:8]}]: {e}")
            return "", True

    def stats(self) -> dict:
        uptime  = round(time.time() - self._start_time, 1)
        recent  = self._batch_stats[-10:] if self._batch_stats else []
        avg_batch   = round(sum(s.batch_size for s in recent) / len(recent), 1) if recent else 0
        avg_iter_ms = round(sum(s.elapsed_ms for s in recent) / len(recent), 1) if recent else 0
        throughput  = round(self._total_tokens_generated / max(uptime, 1), 1)
        return {
            "uptime_seconds":         uptime,
            "total_iterations":       self._total_iterations,
            "total_tokens_generated": self._total_tokens_generated,
            "throughput_tok_per_s":   throughput,
            "avg_batch_size":         avg_batch,
            "avg_iter_ms":            avg_iter_ms,
            "currently_running":      len(self._running),
            "waiting":                self._waiting.qsize(),
            "completed":              len(self._done),
            "max_batch_size":         self.max_batch_size,
        }


# Global singleton
batcher = ContinuousBatcher(max_batch_size=4, max_tokens_per_batch=2048)
