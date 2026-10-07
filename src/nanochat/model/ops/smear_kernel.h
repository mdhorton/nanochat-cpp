// Fused smear (no torch headers, so nvcc stays out of libtorch). Per token t >= 1 of each sequence:
//   gate = lambda * sigmoid(x[t, :gate_cols] . w);  out[t] = x[t] + gate * x[t - 1];  out[0] = x[0]
// The forward rounds as gpt.py's bf16 ops (w and lambda cast to bf16, the gate Linear's output, sigmoid, each product
// and the sum); the backward is fp32 inside, rounded once.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

inline constexpr int kSmearMaxCols = 2048, kSmearMaxGateCols = 256;

// x, out: (rows, cols) bf16 contiguous, 16-byte aligned, rows = B * seq_len in (B, T) order; cols % 8 == 0 and <=
// kSmearMaxCols; gate_cols % 8 == 0 and <= kSmearMaxGateCols. w: gate_cols fp32, lambda: 1 fp32 (device). sig[row]:
// the rounded sigmoid, for backward (0 at t = 0).
void smear_fwd(
      const void* x, const float* w, const float* lambda, void* out, float* sig, int64_t rows, int64_t seq_len,
      int cols, int gate_cols, cudaStream_t stream);

// Blocks the backward uses (its partials buffer holds gate_cols + 1 floats per block).
int smear_bwd_blocks(int64_t rows);

// dx (bf16) from g (rows, cols) bf16 and forward's x, sig; dw (gate_cols) and dlambda (1) fp32, summed in a fixed
// order (deterministic).
void smear_bwd(
      const void* g, const void* x, const float* sig, const float* w, const float* lambda, void* dx, float* partials,
      float* dw, float* dlambda, int64_t rows, int64_t seq_len, int cols, int gate_cols, cudaStream_t stream);

} // namespace nanochat::kernels
