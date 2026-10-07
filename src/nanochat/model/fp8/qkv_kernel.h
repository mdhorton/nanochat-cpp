// Sum of the merged q/k/v input gradients (no torch headers, so nvcc stays out of libtorch).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// dx = dq + dk + dv, plus gate (rows, gate_cols) in the first gate_cols columns when non-null; fp32 sum, rounded once.
// All bf16, contiguous, 16-byte aligned; (rows, cols), cols % 8 == 0.
void qkv_grad_sum(
      const void* dq, const void* dk, const void* dv, const void* gate, int gate_cols, void* dx, int64_t rows, int cols,
      cudaStream_t stream);

} // namespace nanochat::kernels
