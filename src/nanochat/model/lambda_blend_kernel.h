// Fused per-layer residual blend x = resid_lambda * x + x0_lambda * x0 (no torch headers, so nvcc stays out of
// libtorch).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// Blocks the backward uses (the length of its partial sums buffer: 2 floats per block).
int lambda_blend_bwd_blocks(int64_t n);

// x, x0, out: n bf16 values, 16-byte aligned. lr, l0: the lambdas (fp32, device). Rounds as the op-by-op path.
void lambda_blend_fwd(
      const void* x, const void* x0, const float* lr, const float* l0, void* out, int64_t n, cudaStream_t stream);

// dx = lr * g, dx0 = l0 * g; dlr[layer] = sum(g * x), dl0[layer] = sum(g * x0) and 0 at the other n_layer - 1
// entries (a deterministic two-pass sum). partials: 2 * lambda_blend_bwd_blocks(n) floats of scratch.
void lambda_blend_bwd(
      const void* g, const void* x, const void* x0, const float* lr, const float* l0, void* dx, void* dx0,
      float* partials, float* dlr, float* dl0, int layer, int n_layer, int64_t n, cudaStream_t stream);

} // namespace nanochat::kernels
