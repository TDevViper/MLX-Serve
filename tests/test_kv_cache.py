"""Unit tests for the KV cache manager."""
import sys
sys.path.insert(0, ".")

from engine.kv_cache import BlockAllocator, KVCacheManager, BlockStatus

def test_allocate_and_free():
    alloc = BlockAllocator(num_blocks=10, block_size=16)
    assert alloc.num_free_blocks() == 10

    bid = alloc.allocate()
    assert bid is not None
    assert alloc.num_free_blocks() == 9
    assert alloc.num_used_blocks() == 1

    alloc.free(bid)
    assert alloc.num_free_blocks() == 10
    assert alloc.num_used_blocks() == 0
    print("✅ test_allocate_and_free")

def test_oom():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    b1 = alloc.allocate()
    b2 = alloc.allocate()
    b3 = alloc.allocate()  # should be None
    assert b1 is not None
    assert b2 is not None
    assert b3 is None
    print("✅ test_oom")

def test_lru_eviction():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    import time
    b1 = alloc.allocate()
    time.sleep(0.01)
    b2 = alloc.allocate()
    # b1 is LRU
    evicted = alloc.evict_lru()
    assert evicted == b1
    assert alloc.num_free_blocks() == 1
    print("✅ test_lru_eviction")

def test_sequence_lifecycle():
    mgr = KVCacheManager(num_blocks=64, block_size=16)

    # Init with 20 token prompt → needs 2 blocks (ceil(20/16))
    ok = mgr.init_sequence("seq1", prompt_len=20)
    assert ok
    s = mgr.sequence_stats("seq1")
    assert s["num_tokens"] == 20
    assert s["num_blocks"] == 2
    print(f"   seq1: {s['num_tokens']} tokens, {s['num_blocks']} blocks")

    # Generate 10 more tokens
    for _ in range(10):
        ok = mgr.append_token("seq1")
        assert ok

    s = mgr.sequence_stats("seq1")
    assert s["num_tokens"] == 30
    print(f"   after generation: {s['num_tokens']} tokens, {s['num_blocks']} blocks")

    # Free
    mgr.free_sequence("seq1")
    assert mgr.sequence_stats("seq1") == {}
    assert mgr.allocator.num_free_blocks() == 64
    print("✅ test_sequence_lifecycle")

def test_multiple_sequences():
    mgr = KVCacheManager(num_blocks=64, block_size=16)
    for i in range(5):
        ok = mgr.init_sequence(f"seq{i}", prompt_len=32)
        assert ok
    stats = mgr.stats()
    assert stats["active_sequences"] == 5
    print(f"   5 sequences, {stats['used_blocks']} blocks used, {stats['free_blocks']} free")
    print("✅ test_multiple_sequences")

if __name__ == "__main__":
    test_allocate_and_free()
    test_oom()
    test_lru_eviction()
    test_sequence_lifecycle()
    test_multiple_sequences()
    print("\n✅ All KV cache tests passed")
