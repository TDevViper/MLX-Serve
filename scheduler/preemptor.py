"""
Preemptor — evicts lowest-priority sequences when memory is RED.

Priority order (lowest priority evicted first):
  1. Longest waiting sequences (been in queue forever)
  2. Sequences with most tokens (using most KV cache)
  3. Sequences with lowest request_id (oldest)
"""

import asyncio
import logging
from typing import TYPE_CHECKING
from engine.kv_cache import KVCacheManager

if TYPE_CHECKING:
    from scheduler.queue import RequestQueue

logger = logging.getLogger(__name__)


class Preemptor:
    def __init__(self, kv_mgr: KVCacheManager):
        self.kv_mgr   = kv_mgr
        self._queue   = None
        self._monitor = None

    def attach(self, queue, monitor):
        self._queue   = queue
        self._monitor = monitor
        monitor.on_red(self._handle_red_pressure)
        monitor.on_yellow(self._handle_yellow_pressure)
        logger.info("Preemptor attached to memory monitor")

    async def _handle_yellow_pressure(self, snap):
        logger.warning(
            f"⚠️  Memory YELLOW: {snap.utilization*100:.1f}% used "
            f"({snap.free_blocks} blocks free) — monitoring"
        )

    async def _handle_red_pressure(self, snap):
        logger.error(
            f"🔴 Memory RED: {snap.utilization*100:.1f}% used "
            f"({snap.free_blocks} blocks free) — triggering preemption"
        )
        await self._preempt_one()

    async def _preempt_one(self):
        """
        Find the lowest-priority active sequence and free its KV cache blocks.
        The request will be re-queued to retry when memory frees up.
        """
        if self._queue is None:
            return

        # Find active sequences sorted by priority (lowest first)
        active = list(self._queue._active.values())
        if not active:
            logger.info("No active sequences to preempt")
            return

        # Sort by number of KV tokens used (descending = most expensive first)
        def priority_score(req):
            seq_stats = self.kv_mgr.sequence_stats(req.request_id)
            return seq_stats.get("num_tokens", 0)

        victim = max(active, key=priority_score)
        seq_stats = self.kv_mgr.sequence_stats(victim.request_id)

        logger.warning(
            f"Preempting [{victim.request_id[:8]}] — "
            f"{seq_stats.get('num_tokens', 0)} tokens, "
            f"{seq_stats.get('num_blocks', 0)} blocks freed"
        )

        # Free KV cache blocks
        self.kv_mgr.free_sequence(victim.request_id)

        if self._monitor:
            self._monitor.record_preemption()

        logger.info(f"Preemption complete — freed {seq_stats.get('num_blocks', 0)} blocks")
