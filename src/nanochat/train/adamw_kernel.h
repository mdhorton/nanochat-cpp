// Fused AdamW step (no torch headers), op for op as nanochat's adamw_step_fused: fp32 math with torch's rounding at
// every op, the param, grad and state read and written in their own dtype (fp32 or bf16).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// the step's fp32 scalars, computed on the CPU as the op path does
struct AdamWScalars {
  float decay;     // 1 - lr * wd
  float w1, w2;    // 1 - beta1, 1 - beta2 (lerp weights)
  float inv_bias2; // 1 / (1 - beta2^step), as torch's division by a CPU scalar
  float eps;
  float neg_step_size; // -lr / (1 - beta1^step)
};

// p, exp_avg, exp_avg_sq: n contiguous fp32 (bf16_param false) or bf16; grad: n contiguous fp32 or bf16.
void adamw_step(
      void* p, const void* grad, void* exp_avg, void* exp_avg_sq, int64_t n, bool bf16_param, bool bf16_grad,
      const AdamWScalars& s, cudaStream_t stream);

} // namespace nanochat::kernels
