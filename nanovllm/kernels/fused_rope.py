"""
Fused RoPE Triton kernel (in-place version).

Applies rotary embeddings directly to Q and K tensors with zero allocations.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_inplace_kernel(
    Q_ptr,
    K_ptr,
    COS_SIN_ptr,
    POS_ptr,
    stride_qn, stride_qh,
    stride_kn, stride_kh,
    stride_cs_n,
    num_q_heads,
    num_kv_heads,
    HALF_DIM: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """1D grid: (num_tokens,). Modifies Q and K in-place."""
    token_id = tl.program_id(0)
    out_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    pos = tl.load(POS_ptr + token_id)
    dim_offsets = tl.arange(0, HALF_DIM)
    cs_base = pos * stride_cs_n
    cos = tl.load(COS_SIN_ptr + cs_base + dim_offsets).to(tl.float32)
    sin = tl.load(COS_SIN_ptr + cs_base + HALF_DIM + dim_offsets).to(tl.float32)

    # Process all Q heads in-place
    q_base = Q_ptr + token_id * stride_qn
    for h in range(num_q_heads):
        off = h * stride_qh
        x1 = tl.load(q_base + off + dim_offsets).to(tl.float32)
        x2 = tl.load(q_base + off + HALF_DIM + dim_offsets).to(tl.float32)
        tl.store(q_base + off + dim_offsets, (x1 * cos - x2 * sin).to(out_dtype))
        tl.store(q_base + off + HALF_DIM + dim_offsets, (x2 * cos + x1 * sin).to(out_dtype))

    # Process all K heads in-place
    k_base = K_ptr + token_id * stride_kn
    for h in range(num_kv_heads):
        off = h * stride_kh
        x1 = tl.load(k_base + off + dim_offsets).to(tl.float32)
        x2 = tl.load(k_base + off + HALF_DIM + dim_offsets).to(tl.float32)
        tl.store(k_base + off + dim_offsets, (x1 * cos - x2 * sin).to(out_dtype))
        tl.store(k_base + off + HALF_DIM + dim_offsets, (x2 * cos + x1 * sin).to(out_dtype))


def fused_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings in-place. Zero allocations."""
    N = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    half_dim = query.shape[2] // 2
    is_bf16 = query.dtype == torch.bfloat16

    _fused_rope_inplace_kernel[(N,)](
        query, key, cos_sin_cache, positions,
        query.stride(0), query.stride(1),
        key.stride(0), key.stride(1),
        cos_sin_cache.stride(0),
        num_q_heads, num_kv_heads,
        HALF_DIM=half_dim,
        IS_BF16=is_bf16,
        num_warps=4,
    )
    return query, key
