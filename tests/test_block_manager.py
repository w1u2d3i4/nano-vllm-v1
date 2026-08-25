from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


BLOCK_SIZE = 256


def make_sequence(num_tokens: int) -> Sequence:
    return Sequence(
        list(range(num_tokens)),
        SamplingParams(max_tokens=8, ignore_eos=True),
    )


def test_decode_allocates_at_257_token_boundary() -> None:
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    seq = make_sequence(BLOCK_SIZE)
    manager.allocate(seq, num_cached_blocks=0)

    # Prefill caches 256 prompt tokens, then sampling appends token 257.
    seq.num_cached_tokens = BLOCK_SIZE
    seq.append_token(10_000)

    assert len(seq) == BLOCK_SIZE + 1
    assert manager.can_append(seq)
    manager.may_append(seq)
    assert len(seq.block_table) == 2


def test_decode_reuses_first_block_for_token_256() -> None:
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    seq = make_sequence(BLOCK_SIZE - 1)
    manager.allocate(seq, num_cached_blocks=0)

    # Token 256 is written to the last slot of the existing block.
    seq.num_cached_tokens = BLOCK_SIZE - 1
    seq.append_token(10_000)

    assert len(seq) == BLOCK_SIZE
    manager.may_append(seq)
    assert len(seq.block_table) == 1


def test_prefix_cache_reuses_only_complete_non_tail_blocks() -> None:
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    first = make_sequence(2 * BLOCK_SIZE)
    manager.allocate(first, num_cached_blocks=0)
    first.num_scheduled_tokens = len(first)
    manager.hash_blocks(first)
    manager.deallocate(first)

    second = make_sequence(2 * BLOCK_SIZE)
    num_cached_blocks = manager.can_allocate(second)

    # The tail block is deliberately recomputed because it can still grow.
    assert num_cached_blocks == 1
    manager.allocate(second, num_cached_blocks)
    assert second.num_cached_tokens == BLOCK_SIZE


def test_hash_collision_does_not_reuse_different_tokens(monkeypatch) -> None:
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    monkeypatch.setattr(manager, "compute_hash", lambda *_args, **_kwargs: 7)

    first = make_sequence(2 * BLOCK_SIZE)
    manager.allocate(first, num_cached_blocks=0)
    first.num_scheduled_tokens = len(first)
    manager.hash_blocks(first)
    manager.deallocate(first)

    second = make_sequence(2 * BLOCK_SIZE)
    second.token_ids[0] = 99_999

    assert manager.can_allocate(second) == 0
