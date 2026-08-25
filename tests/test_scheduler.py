from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


BLOCK_SIZE = 256


def make_scheduler(*, num_blocks: int = 8, token_budget: int = BLOCK_SIZE) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_num_seqs = 8
    scheduler.max_num_batched_tokens = token_budget
    scheduler.eos = -1
    scheduler.block_size = BLOCK_SIZE
    scheduler.block_manager = BlockManager(num_blocks, BLOCK_SIZE)
    scheduler.waiting = deque()
    scheduler.running = deque()
    return scheduler


def make_sequence(num_tokens: int, *, max_tokens: int = 8) -> Sequence:
    return Sequence(
        list(range(num_tokens)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )


def test_chunked_prefill_crosses_block_boundary_then_decodes() -> None:
    scheduler = make_scheduler(num_blocks=8, token_budget=BLOCK_SIZE)
    seq = make_sequence(2 * BLOCK_SIZE)
    scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill
    assert scheduled == [seq]
    assert seq.num_scheduled_tokens == BLOCK_SIZE
    scheduler.postprocess(scheduled, [90_000], is_prefill)
    assert seq.status is SequenceStatus.WAITING
    assert seq.num_cached_tokens == BLOCK_SIZE
    assert len(seq) == 2 * BLOCK_SIZE

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill
    assert seq.num_scheduled_tokens == BLOCK_SIZE
    scheduler.postprocess(scheduled, [90_001], is_prefill)
    assert seq.status is SequenceStatus.RUNNING
    assert seq.num_cached_tokens == 2 * BLOCK_SIZE
    assert len(seq) == 2 * BLOCK_SIZE + 1

    scheduled, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert scheduled == [seq]
    assert seq.num_scheduled_tokens == 1
    assert len(seq.block_table) == 3


def test_decode_preempts_when_boundary_needs_a_block() -> None:
    scheduler = make_scheduler(num_blocks=2, token_budget=BLOCK_SIZE)
    seq = make_sequence(BLOCK_SIZE)
    scheduler.block_manager.allocate(seq, num_cached_blocks=0)
    seq.num_cached_tokens = BLOCK_SIZE
    seq.status = SequenceStatus.RUNNING
    seq.append_token(90_000)

    victim = make_sequence(BLOCK_SIZE // 2)
    scheduler.block_manager.allocate(victim, num_cached_blocks=0)
    victim.num_cached_tokens = len(victim)
    victim.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    scheduler.running.append(victim)

    scheduled, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert scheduled == [seq]
    assert seq.status is SequenceStatus.RUNNING
    assert len(seq.block_table) == 2
    assert victim.status is SequenceStatus.WAITING
    assert list(scheduler.waiting) == [victim]


def test_finished_sequence_releases_all_blocks() -> None:
    scheduler = make_scheduler(num_blocks=4, token_budget=BLOCK_SIZE)
    seq = make_sequence(BLOCK_SIZE, max_tokens=1)
    scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [90_000], is_prefill)

    assert seq.status is SequenceStatus.FINISHED
    assert not seq.block_table
    assert len(scheduler.block_manager.free_block_ids) == 4
    assert scheduler.is_finished()
