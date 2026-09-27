// Fused softcap + cross-entropy over bf16 logits (no torch headers, so nvcc stays out of libtorch).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// logits: (rows, ld) bf16, columns [0, padded) used, padded % 8 == 0 and ld % 8 == 0. Writes loss[row] (if
// non-null) = lse - capped[target], 0 for target < 0. If grad, overwrites logits with d loss_row / d logits
// scaled by grad_scale[row * stride] (null = 1) and 1 / *num_valid (null = 1); columns >= vocab get 0. grad_amax
// (device float, may be null): set to the max |gradient| (as bf16), for its FP8 quantization.
void softcap_ce(
      void* logits, int64_t rows, int64_t ld, const int64_t* targets, int vocab, int padded, float softcap, float* loss,
      const float* grad_scale, int64_t grad_scale_stride, const int64_t* num_valid, bool grad, float* grad_amax,
      cudaStream_t stream);

} // namespace nanochat::kernels
