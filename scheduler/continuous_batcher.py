"""
Continuous Batching Scheduler for MLX-Serve.

At every generation step, the scheduler:
1. Checks waiting queue for new requests that fit in memory
2. Runs one forward pass for ALL active sequences together
3. Removes finished sequences, adds new ones
4. Repeats

This is iteration-level scheduling — the key insight of the
"Orca" paper (2022) that vLLM is built on.
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
    """A single inference sequence being tracked by the batcher."""
    seq_id:       str
    prompt:       str
    prompt_tokens: int
    max_tokens:   int
    temperature:  float
    stream:       bool
    created_at:   float = field(default_factory=time.time)
    status:       SeqStatus = SeqStatus.WAITING

    # Generation state
    generated_tokens: int = 0
    output_text:      str = ""
    finish_reason:    str = ""

    # For streaming
    token_queue: Optional[asyncio.Queue] = field(default=None, repr=False)

    # Timing
    start_time:  Optional[float] = None
    finish_time: Optional[float] = None

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.finish_time or time.time()
        return round(end - self.start_time, 3)

    def tokens_per_sec(self) -> float:
        e = self.elapsed()
        if e == 0 or self.generated_tokens == 0:
            return 0.0
        return round(self.generated_tokens / e, 1)

    def is_finished(self) -> bool:
        return self.generated_tokens >= self.max_tokens

    def __post_init__(self):
        if self.stream:
            self.token_queue = asyncio.Queue()


@dataclass
class BatchStats:
    """Stats for one iteration of the continuous batcher."""
    iteration:        int
    batch_size:       int
    waiting:          int
    tokens_generated: int
    elapsed_ms:       float
    timestamp:        float = field(default_factory=time.time)


class ContinuousBatcher:
    """
    Iteration-level scheduler that processes multiple sequences per step.

    Key properties:
    - max_batch_size: max sequences in one forward pass
    - max_tokens_per_batch: memory budget per iteration
    - Sequences join/leave the batch between iterations
    """

    def __init__(
        self,
        max_batch_size: int = 4,
        max_tokens_per_batch: int = 2048,
        scheduler_interval: float = 0.0,  # 0 = run as fast as possible
    ):
        self.max_batch_size       = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.scheduler_interval   = scheduler_interval

        self._waiting:  asyncio.Queue  = asyncio.Queue()
        self._running:  Dict[str, Sequence] = {}
        self._done:     Dict[str, Sequence] = {}

        self._running_task: Optional[asyncio.Task] = None
        self._active = False

        # Stats
        self._total_iterations = 0
        self._total_tokens_generated = 0
        self._batch_stats: List[BatchStats] = []
        self._max_stats = 100
        self._start_time = time.time()

    async def start(self):
        self._active = True
        self._running_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            f"Continuous batcher started — "
            f"max_batch={self.max_batch_size}, "
            f"max_tokens_per_batch={self.max_tokens_per_batch}"
        )

    async def stop(self):
        self._active = False
        if self._running_task:
            self._running_task.cancel()

    async def submit(self, seq: Sequence):
        """Add a sequence to the waiting queue."""
        await self._waiting.put(seq)
        logger.debug(f"Submitted [{seq.seq_id[:8]}] — waiting: {self._waiting.qsize()}")

    async def wait_for_result(self, seq_id: str, timeout: float = 120.0) -> Optional[Sequence]:
        """Block until a sequence is done."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if seq_id in self._done:
                return self._done[seq_id]
            await asyncio.sleep(0.05)
        return None

    async def _scheduler_loop(self):
        """Main scheduler loop — runs continuously."""
        from engine.model_runner import runner

        while self._active:
            # ── Step 1: admit waiting sequences into running batch ──────────
            await self._admit_sequences()

            if not self._running:
                await asyncio.sleep(0.01)
                continue

            # ── Step 2: run one generation step for all active sequences ────
            t0 = time.time()
            tokens_this_iter = await self._run_one_step(runner)
            elapsed_ms = (time.time() - t0) * 1000

            # ── Step 3: record stats ────────────────────────────────────────
            self._total_iterations += 1
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

    async def _admit_sequences(self):
        """Move waiting sequences into the running batch if there's capacity."""
        while (
            len(self._running) < self.max_batch_size
            and not self._waiting.empty()
        ):
            try:
                seq = self._waiting.get_nowait()
                seq.status = SeqStatus.RUNNING
                seq.start_time = time.time()
                self._running[seq.seq_id] = seq
                logger.debug(
                    f"Admitted [{seq.seq_id[:8]}] — "
                    f"batch size now {len(self._running)}"
                )
            except asyncio.QueueEmpty:
                break

    async def _run_one_step(self, runner) -> int:
        """
        Run one generation step for all sequences in the running batch.

        In a true continuous batching implementation this would be a single
        batched forward pass. Since MLX generate() is per-sequence, we run
        each sequence for one token using stream_generate and yield control.

        Returns: number of tokens generated this step.
        """
        loop = asyncio.get_event_loop()
        finished = []
        tokens_generated = 0

        for seq_id, seq in list(self._running.items()):
            # Generate one token for this sequence
            token = await loop.run_in_executor(
                None, self._generate_one_token, seq, runner
            )

            if token is not None:
                seq.output_text += token
                seq.generated_tokens += 1
                tokens_generated += 1

                # Push to stream queue if streaming
                if seq.stream and seq.token_queue:
                    await seq.token_queue.put(token)

            # Check if sequence is done
            if seq.is_finished() or token is None:
                seq.status = SeqStatus.DONE
                seq.finish_time = time.time()
                if seq.stream and seq.token_queue:
                    await seq.token_queue.put(None)  # signal done
                finished.append(seq_id)
                logger.debug(
                    f"Finished [{seq_id[:8]}] — "
                    f"{seq.generated_tokens} tokens, "
                    f"{seq.tokens_per_sec()} tok/s"
                )

        # Move finished sequences out of running
        for seq_id in finished:
            seq = self._running.pop(seq_id)
            self._done[seq_id] = seq

        return tokens_generated

    def _generate_one_token(self, seq: Sequence, runner) -> Optional[str]:
        """
        Generate exactly one token for a sequence using stream_generate.
        Returns the token text, or None if generation is complete.
        """
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Build the full prompt including any already-generated text
        full_prompt = seq.prompt + seq.output_text
        sampler = make_sampler(seq.temperature)

        try:
            for response in stream_generate(
                runner.model,
                runner.tokenizer,
                prompt=full_prompt,
                max_tokens=1,  # one token at a time
                sampler=sampler,
            ):
                return response.text
        except Exception as e:
            logger.error(f"Generation error for [{seq.seq_id[:8]}]: {e}")
            return None
        return None

    def stats(self) -> dict:
        uptime = round(time.time() - self._start_time, 1)
        recent = self._batch_stats[-10:] if self._batch_stats else []
        avg_batch = (
            round(sum(s.batch_size for s in recent) / len(recent), 1)
            if recent else 0
        )
        avg_iter_ms = (
            round(sum(s.elapsed_ms for s in recent) / len(recent), 1)
            if recent else 0
        )
        throughput = (
            round(self._total_tokens_generated / max(uptime, 1), 1)
        )
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
