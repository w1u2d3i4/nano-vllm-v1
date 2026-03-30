import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        if config.enable_ltr:
            from nanovllm.engine.ltr_scheduler import LTRScheduler
            self.scheduler = LTRScheduler(config)
        else:
            self.scheduler = Scheduler(config)
        atexit.register(self.exit)

        # Speculative decoding config
        self.enable_speculative = config.enable_speculative
        self.num_draft_layers = config.num_draft_layers
        self.num_speculative_tokens = config.num_speculative_tokens

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        if hasattr(self.scheduler, "save_state"):
            self.scheduler.save_state()
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs = self.scheduler.schedule()

        # Try speculative decode for pure decode batches
        if self.enable_speculative and self._can_speculate(seqs):
            return self._speculative_step(seqs)

        # Normal step
        token_ids, seq_need_compute_logits = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_total_tokens = sum(len(seq) for seq in seqs if seq.is_finished)
        return outputs, num_total_tokens

    def _can_speculate(self, seqs):
        """Check if all seqs are in pure decode mode."""
        if not seqs:
            return False
        for seq in seqs:
            if not seq.block_table or seq.num_new_tokens != 1:
                return False
        return True

    @torch.inference_mode()
    def _speculative_step(self, seqs):
        """
        Speculative decoding step.

        Flow:
        1. Draft: K iterations with early-exit model, generating K tokens
        2. Verify: run full model on (pending_token + K draft tokens) = K+1 tokens
           - verify_logits[j] verifies draft_token[j]
           - verify_logits[K] gives bonus token
        3. Rejection sampling → accept 1 to K+1 tokens per seq
        """
        from nanovllm.engine.speculative import verify_and_accept

        K = self.num_speculative_tokens
        num_draft_layers = self.num_draft_layers
        scheduler = self.scheduler
        block_manager = scheduler.block_manager
        model_runner = self.model_runner

        # Save state before draft
        # At entry: num_cached_tokens = T, num_new_tokens = 1 (pending token at position T)
        orig_cached = {seq.seq_id: seq.num_cached_tokens for seq in seqs}

        # ── Draft phase: K iterations ──
        # Iteration 0 processes the pending token; iterations 1..K-1 process draft tokens
        all_draft_tokens = {seq.seq_id: [] for seq in seqs}
        all_draft_probs = {seq.seq_id: [] for seq in seqs}

        def _ensure_blocks(seq, num_tokens):
            """Ensure seq has enough blocks for num_tokens total context, without hash checks."""
            needed = (num_tokens + block_manager.block_size - 1) // block_manager.block_size
            while len(seq.block_table) < needed:
                block_id = block_manager.free_block_ids[0]
                block_manager._allocate_block(block_id)
                seq.block_table.append(block_id)

        for k in range(K):
            for seq in seqs:
                seq.num_new_tokens = 1
                _ensure_blocks(seq, seq.num_cached_tokens + 1)

            draft_token_ids, draft_probs = model_runner.run_draft(seqs, num_draft_layers)

            for idx, seq in enumerate(seqs):
                token_id = draft_token_ids[idx]
                all_draft_tokens[seq.seq_id].append(token_id)
                all_draft_probs[seq.seq_id].append(draft_probs[idx])
                seq.append_token(token_id)
                seq.num_cached_tokens += 1

        # After draft: seq has K new tokens (d_0..d_{K-1})
        # num_tokens = orig + K, num_cached_tokens = orig_cached + K

        # ── Prepare verify ──
        # Roll back cache to process pending + K drafts = K+1 tokens with full model
        for seq in seqs:
            seq.num_cached_tokens = orig_cached[seq.seq_id]
            seq.num_new_tokens = K + 1  # pending + K draft tokens
            _ensure_blocks(seq, seq.num_cached_tokens + K + 1)

        # ── Verify: full model on K+1 tokens ──
        verify_logits = model_runner.run_verify(seqs)
        # verify_logits: (num_seqs * (K+1), vocab_size)

        # Build per-seq data
        draft_probs_list = []
        draft_tokens_list = []
        temperatures = []
        for seq in seqs:
            draft_probs_list.append(torch.stack(all_draft_probs[seq.seq_id]))
            draft_tokens_list.append(all_draft_tokens[seq.seq_id])
            temperatures.append(seq.temperature)

        # ── Rejection sampling ──
        results = verify_and_accept(
            verify_logits, draft_probs_list, draft_tokens_list,
            temperatures, len(seqs), K,
        )

        # ── Apply results ──
        outputs = []

        for idx, seq in enumerate(seqs):
            accepted_tokens, num_from_draft = results[idx]

            # Roll back ALL draft tokens
            seq.rollback(K)
            seq.num_cached_tokens = orig_cached[seq.seq_id]

            # Append accepted tokens
            for token_id in accepted_tokens:
                seq.append_token(token_id)

            # Verify wrote KV for K+1 positions (pending + K drafts).
            # The pending token's KV is always valid.
            # For accepted draft tokens, their KV from verify is correct.
            # num_from_draft = number of draft tokens accepted (0..K)
            # +1 for the pending token
            seq.num_cached_tokens = orig_cached[seq.seq_id] + 1 + num_from_draft
            seq.num_new_tokens = 0

            # Trim excess blocks
            block_manager.trim_blocks(seq)

            # Check termination for each accepted token
            for token_id in accepted_tokens:
                if (not seq.ignore_eos and token_id == scheduler.eos) or \
                    seq.num_completion_tokens >= seq.max_tokens or \
                        len(seq) >= scheduler.max_model_len:
                    seq.status = SequenceStatus.FINISHED
                    block_manager.deallocate(seq)
                    scheduler.running.remove(seq)
                    break

            if seq.is_finished:
                outputs.append((seq.seq_id, seq.completion_token_ids))

        num_total_tokens = sum(len(seq) for seq in seqs if seq.is_finished)
        return outputs, num_total_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        num_total_tokens = 0
        t = perf_counter()
        while not self.is_finished():
            output, num_step_tokens = self.step()
            num_total_tokens += num_step_tokens
            if use_tqdm:
                total_throughput = num_total_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "total_throughput": f"{int(total_throughput)}tok/s",
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
