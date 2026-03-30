"""
Speculative decoding with layer-skip draft model.

Rejection sampling (Leviathan 2023) guarantees identical output distribution.
"""
import torch


def rejection_sample(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    bonus_probs: torch.Tensor,
) -> tuple[list[int], int]:
    """
    Rejection sampling for one sequence.

    Args:
        target_probs: (K, vocab_size) — target probs for verifying each draft token
        draft_probs:  (K, vocab_size) — draft probs
        draft_tokens: (K,) — draft token ids
        bonus_probs:  (vocab_size,) — target probs for bonus token (if all accepted)

    Returns:
        (accepted_tokens, num_from_draft)
    """
    K = draft_tokens.shape[0]
    accepted = []

    for i in range(K):
        x_i = draft_tokens[i].item()
        p = target_probs[i, x_i].item()
        q = draft_probs[i, x_i].item()

        r = torch.rand(1).item()
        if q > 0 and r < min(1.0, p / q):
            accepted.append(x_i)
        else:
            # Resample from adjusted distribution
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            adj_sum = adjusted.sum()
            if adj_sum > 0:
                adjusted = adjusted / adj_sum
            else:
                adjusted = target_probs[i]
            x_new = torch.multinomial(adjusted, 1).item()
            accepted.append(x_new)
            return accepted, i

    # All K accepted: add bonus token
    bonus_token = torch.multinomial(bonus_probs, 1).item()
    accepted.append(bonus_token)
    return accepted, K


def verify_and_accept(
    verify_logits: torch.Tensor,
    draft_probs_list: list[torch.Tensor],
    draft_tokens_list: list[list[int]],
    temperatures: list[float],
    num_seqs: int,
    K: int,
) -> list[tuple[list[int], int]]:
    """
    Batch rejection sampling.

    verify_logits: (num_seqs * (K+1), vocab_size) — logits from verify forward
        For each seq: K+1 positions [pending, d_0, ..., d_{K-1}]
        logits[j] at position j → p(next | context up to j) → verifies d_j
        logits[K] → bonus token

    Returns: list of (accepted_tokens, num_from_draft) per sequence
    """
    results = []
    offset = 0

    for i in range(num_seqs):
        seq_logits = verify_logits[offset:offset + K + 1]  # (K+1, vocab_size)
        offset += K + 1

        temp = temperatures[i]

        # Compute target probs with temperature
        target_logits = seq_logits[:K].float()  # (K, vocab_size) — for verifying d_0..d_{K-1}
        if temp > 0:
            target_logits = target_logits / temp
        target_probs = torch.softmax(target_logits, dim=-1)

        bonus_logits = seq_logits[K].float()  # (vocab_size,) — for bonus
        if temp > 0:
            bonus_logits = bonus_logits / temp
        bonus_probs = torch.softmax(bonus_logits, dim=-1)

        draft_probs = draft_probs_list[i]  # (K, vocab_size)
        draft_tokens = torch.tensor(draft_tokens_list[i], device=verify_logits.device)

        accepted, num_from_draft = rejection_sample(
            target_probs, draft_probs, draft_tokens, bonus_probs)
        results.append((accepted, num_from_draft))

    return results
