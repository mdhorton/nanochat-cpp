// Fused rotary embedding + RMSNorm + scale for q/k heads (no torch headers, so nvcc stays out of libtorch).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// x, out: (rows, head_dim) bf16, rows ordered as (B, T, heads); row r is at position t = (r / heads) % seq_len.
// cos, sin: (seq_len, head_dim / 2) bf16. head_dim % 4 == 0, head_dim <= 512.
// out = scale * rms_norm(rotary(x)) (as gpt.py's apply_rotary_emb, then norm), computed in fp32;
// rstd[row] = 1 / sqrt(mean(rotary(x)^2) + eps).
void rotary_norm_fwd(
      const void* x, const void* cos, const void* sin, void* out, float* rstd, int64_t rows, int heads, int64_t seq_len,
      int head_dim, float scale, float eps, cudaStream_t stream);

// dx from dout, given forward's x and rstd.
void rotary_norm_bwd(
      const void* dout, const void* x, const void* cos, const void* sin, const float* rstd, void* dx, int64_t rows,
      int heads, int64_t seq_len, int head_dim, float scale, cudaStream_t stream);

} // namespace nanochat::kernels
