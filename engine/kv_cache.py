"""
PagedAttention-style KV Cache Manager for MLX-Serve.

Key concepts:
- Block: fixed-size unit of KV cache memory (default 16 tokens)
- BlockTable: maps logical block indices to physical block indices per sequence
- BlockAllocator: manages the pool of free/used physical blocks
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class BlockStatus(Enum):
    FREE     = "free"
    USED     = "used"
    RESERVED = "reserved"


@dataclass
class PhysicalBlock:
    """A fixed-size block of KV cache memory."""
    block_id:    int
    status:      BlockStatus = BlockStatus.FREE
    ref_count:   int = 0          # how many sequences share this block
    last_access: float = field(default_factory=time.time)
    token_ids:   List[int] = field(default_factory=list)  # for prefix caching (Week 9)

    def is_free(self) -> bool:
        return self.status == BlockStatus.FREE and self.ref_count == 0


@dataclass
class LogicalBlock:
    """Maps a sequence position range to a physical block."""
    logical_idx:  int
    physical_idx: Optional[int] = None
    num_tokens:   int = 0

    def is_full(self, block_size: int) -> bool:
        return self.num_tokens >= block_size


class BlockAllocator:
    """
    Manages a pool of physical KV cache blocks.
    Thread-safe — multiple coroutines can allocate/free concurrently.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._lock = threading.Lock()

        # Initialize all blocks as free
        self._blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i) for i in range(num_blocks)
        }
        self._free_blocks: Set[int] = set(range(num_blocks))
        self._used_blocks: Set[int] = set()

        # Stats
        self._total_allocations = 0
        self._total_frees = 0
        self._evictions = 0

    def allocate(self) -> Optional[int]:
        """Allocate a free block. Returns block_id or None if OOM."""
        with self._lock:
            if not self._free_blocks:
                return None  # OOM — caller should handle (evict or queue)
            block_id = self._free_blocks.pop()
            block = self._blocks[block_id]
            block.status = BlockStatus.USED
            block.ref_count = 1
            block.last_access = time.time()
            block.token_ids = []
            self._used_blocks.add(block_id)
            self._total_allocations += 1
            return block_id

    def free(self, block_id: int) -> bool:
        """Free a block back to the pool."""
        with self._lock:
            if block_id not in self._blocks:
                return False
            block = self._blocks[block_id]
            block.ref_count -= 1
            if block.ref_count <= 0:
                block.status = BlockStatus.FREE
                block.ref_count = 0
                block.token_ids = []
                self._used_blocks.discard(block_id)
                self._free_blocks.add(block_id)
                self._total_frees += 1
            return True

    def fork(self, block_id: int) -> int:
        """
        Copy-on-write: increment ref_count for shared blocks.
        Used when a sequence is forked (beam search, parallel sampling).
        """
        with self._lock:
            block = self._blocks[block_id]
            block.ref_count += 1
            return block_id

    def evict_lru(self) -> Optional[int]:
        """Evict the least-recently-used block. Returns freed block_id."""
        with self._lock:
            candidates = [
                self._blocks[bid] for bid in self._used_blocks
                if self._blocks[bid].ref_count == 1  # only evict unshared
            ]
            if not candidates:
                return None
            lru = min(candidates, key=lambda b: b.last_access)
            lru.status = BlockStatus.FREE
            lru.ref_count = 0
            lru.token_ids = []
            self._used_blocks.discard(lru.block_id)
            self._free_blocks.add(lru.block_id)
            self._evictions += 1
            return lru.block_id

    def num_free_blocks(self) -> int:
        with self._lock:
            return len(self._free_blocks)

    def num_used_blocks(self) -> int:
        with self._lock:
            return len(self._used_blocks)

    def utilization(self) -> float:
        return round(len(self._used_blocks) / self.num_blocks, 3)

    def stats(self) -> dict:
        return {
            "total_blocks":       self.num_blocks,
            "free_blocks":        self.num_free_blocks(),
            "used_blocks":        self.num_used_blocks(),
            "utilization":        self.utilization(),
            "block_size_tokens":  self.block_size,
            "total_kv_tokens":    self.num_blocks * self.block_size,
            "used_kv_tokens":     self.num_used_blocks() * self.block_size,
            "total_allocations":  self._total_allocations,
            "total_frees":        self._total_frees,
            "evictions":          self._evictions,
        }


class SequenceBlockTable:
    """
    Tracks which physical blocks are assigned to a single sequence.
    The block table is the mapping: logical_block_idx -> physical_block_id
    """

    def __init__(self, seq_id: str, block_size: int, allocator: BlockAllocator):
        self.seq_id     = seq_id
        self.block_size = block_size
        self.allocator  = allocator
        self._table: List[int] = []       # [logical_idx] -> physical_block_id
        self._num_tokens = 0

    def append_tokens(self, num_new_tokens: int) -> bool:
        """
        Ensure enough blocks are allocated for num_new_tokens more tokens.
        Returns False if allocation failed (OOM).
        """
        for _ in range(num_new_tokens):
            # Check if current last block is full or doesn't exist
            if not self._table or self._is_last_block_full():
                block_id = self.allocator.allocate()
                if block_id is None:
                    # Try eviction
                    evicted = self.allocator.evict_lru()
                    if evicted is None:
                        return False  # truly OOM
                    block_id = evicted
                self._table.append(block_id)
            self._num_tokens += 1
        return True

    def _is_last_block_full(self) -> bool:
        tokens_in_last = self._num_tokens % self.block_size
        return tokens_in_last == 0 and self._num_tokens > 0

    def free_all(self):
        """Release all blocks back to the allocator."""
        for block_id in self._table:
            self.allocator.free(block_id)
        self._table = []
        self._num_tokens = 0

    def physical_blocks(self) -> List[int]:
        return list(self._table)

    def num_blocks(self) -> int:
        return len(self._table)

    def num_tokens(self) -> int:
        return self._num_tokens

    def memory_bytes(self, bytes_per_token: int = 512) -> int:
        """Estimate memory used by this sequence's KV cache."""
        return self._num_tokens * bytes_per_token


class KVCacheManager:
    """
    Top-level KV cache manager.
    Manages block tables for all active sequences.
    """

    def __init__(self, num_blocks: int = 512, block_size: int = 16):
        self.allocator  = BlockAllocator(num_blocks, block_size)
        self.block_size = block_size
        self._sequences: Dict[str, SequenceBlockTable] = {}
        self._lock = threading.Lock()

    def init_sequence(self, seq_id: str, prompt_len: int) -> bool:
        """
        Initialize KV cache for a new sequence.
        Allocates blocks for the prompt tokens.
        """
        with self._lock:
            if seq_id in self._sequences:
                return True
            table = SequenceBlockTable(seq_id, self.block_size, self.allocator)
            ok = table.append_tokens(prompt_len)
            if ok:
                self._sequences[seq_id] = table
            return ok

    def append_token(self, seq_id: str) -> bool:
        """Called each generation step to allocate for one new token."""
        with self._lock:
            if seq_id not in self._sequences:
                return False
            return self._sequences[seq_id].append_tokens(1)

    def free_sequence(self, seq_id: str):
        """Release all KV cache blocks for a completed/cancelled sequence."""
        with self._lock:
            if seq_id in self._sequences:
                self._sequences[seq_id].free_all()
                del self._sequences[seq_id]

    def get_block_table(self, seq_id: str) -> Optional[List[int]]:
        """Get the physical block IDs for a sequence (for attention computation)."""
        with self._lock:
            if seq_id not in self._sequences:
                return None
            return self._sequences[seq_id].physical_blocks()

    def sequence_stats(self, seq_id: str) -> dict:
        with self._lock:
            if seq_id not in self._sequences:
                return {}
            t = self._sequences[seq_id]
            return {
                "seq_id":        seq_id,
                "num_tokens":    t.num_tokens(),
                "num_blocks":    t.num_blocks(),
                "memory_bytes":  t.memory_bytes(),
            }

    def stats(self) -> dict:
        s = self.allocator.stats()
        s["active_sequences"] = len(self._sequences)
        return s


# Global singleton
kv_cache = KVCacheManager(num_blocks=512, block_size=16)
