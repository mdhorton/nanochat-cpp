// Fused residual step (no torch headers, so nvcc stays out of libtorch). Per row of `cols` bf16 values:
//   s = r ? x + r : x;  res = x0 ? lr * s + l0 * x0 : s;  n = rms_norm(res)
// The add and the lambda blend round as the op path (see residual_norm_kernel.cu).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

inline constexpr int kResidualNormMaxCols = 2048;

// All tensors contiguous, 16-byte aligned; cols % 8 == 0, cols <= kResidualNormMaxCols.
struct ResidualNormFwd {
  const void *x, *r, *x0; // r, x0 may be null
  const float *lr, *l0;   // the layer's lambdas (fp32, device), with x0
  void* s;                // x + r, written when both r and x0 are given (the lambda grads need it), else null
  void *res, *n;
  float* rstd;
  int64_t rows;
  int cols;
  float eps;
};

void residual_norm_fwd(const ResidualNormFwd& a, cudaStream_t stream);

// Blocks the backward uses (its partial sums buffer holds 2 floats per block).
int residual_norm_bwd_blocks(int64_t rows);

// g = g_res + rms_norm backward of g_n (each may be null: no gradient). Without x0: ds = g. With x0: ds = lr * g,
// dx0 = l0 * g, dlr[layer] = sum(g * s), dl0[layer] = sum(g * x0), and 0 at the other n_layer - 1 entries.
struct ResidualNormBwd {
  const void *g_res, *g_n;
  const void* res;
  const float* rstd;
  const void *s, *x0; // blend only: s = the blend's input
  const float *lr, *l0;
  void *ds, *dx0;
  float *partials, *dlr, *dl0; // partials: 2 * residual_norm_bwd_blocks(rows) floats of scratch
  int layer, n_layer;
  int64_t rows;
  int cols;
};

void residual_norm_bwd(const ResidualNormBwd& a, cudaStream_t stream);

} // namespace nanochat::kernels
