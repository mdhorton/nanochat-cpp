// Fused backout (no torch headers): out = x - lambda * xb, rounding as gpt.py's bf16 ops (lambda cast to bf16, the
// product, the difference).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// x, xb, out: n contiguous bf16, 16-byte aligned, n % 8 == 0; lambda: 1 fp32 (device).
void backout_fwd(const void* x, const void* xb, const float* lambda, void* out, int64_t n, cudaStream_t stream);

// Blocks the backward uses (its partials buffer holds one float per block).
int backout_bwd_blocks(int64_t n);

// dxb = -lambda * g (bf16, as the ops), dlambda = -sum(g * xb) in fp32, summed in a fixed order (deterministic).
void backout_bwd(
      const void* g, const void* xb, const float* lambda, void* dxb, float* partials, float* dlambda, int64_t n,
      cudaStream_t stream);

} // namespace nanochat::kernels
