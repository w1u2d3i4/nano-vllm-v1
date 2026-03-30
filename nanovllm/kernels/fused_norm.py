"""
Fused Add + RMSNorm Triton kernel.

- rms_forward:     allocates output (cannot be in-place; caller saves original as residual)
- add_rms_forward: in-place — x ← norm(x+res), res ← x+res (zero extra allocations)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    X_ptr,
    OUT_ptr,
    W_ptr,
    stride_n,
    D_actual,
    eps,
    D: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    mask = cols < D_actual
    offset = row * stride_n

    x = tl.load(X_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D_actual
    x_norm = x * tl.rsqrt(var + eps)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out_dtype = tl.bfloat16 if IS_BF16 else tl.float16
    tl.store(OUT_ptr + offset + cols, (x_norm * w).to(out_dtype), mask=mask)


@triton.jit
def _add_rms_norm_inplace_kernel(
    X_ptr,       # in/out: overwritten with norm(x + res)
    RES_ptr,     # in/out: overwritten with (x + res)
    W_ptr,
    stride_n,
    D_actual,
    eps,
    D: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    mask = cols < D_actual
    offset = row * stride_n
    out_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    x = tl.load(X_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(RES_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)

    added = x + res
    tl.store(RES_ptr + offset + cols, added.to(out_dtype), mask=mask)

    var = tl.sum(added * added, axis=0) / D_actual
    normed = added * tl.rsqrt(var + eps)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(X_ptr + offset + cols, (normed * w).to(out_dtype), mask=mask)


def _get_num_warps(D):
    if D <= 128:
        return 2
    elif D <= 512:
        return 4
    elif D <= 2048:
        return 8
    return 16


def rms_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with new output tensor. Supports 2D (N, D) and 3D (N, H, D)."""
    orig_shape = x.shape
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    N, D_actual = x.shape
    D = triton.next_power_of_2(D_actual)
    out = torch.empty_like(x)
    is_bf16 = x.dtype == torch.bfloat16
    _rms_norm_kernel[(N,)](
        x, out, weight,
        x.stride(0), D_actual, eps,
        D=D, IS_BF16=is_bf16,
        num_warps=_get_num_warps(D),
    )
    return out.reshape(orig_shape)


def add_rms_forward(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused add + RMSNorm in-place: x ← norm(x+res), res ← x+res."""
    N, D_actual = x.shape
    D = triton.next_power_of_2(D_actual)
    is_bf16 = x.dtype == torch.bfloat16
    _add_rms_norm_inplace_kernel[(N,)](
        x, residual, weight,
        x.stride(0), D_actual, eps,
        D=D, IS_BF16=is_bf16,
        num_warps=_get_num_warps(D),
    )
    return x, residual
