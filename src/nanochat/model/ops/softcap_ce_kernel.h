// Fused softcap + cross-entropy over bf16 logits (no torch headers, so nvcc stays out of libtorch).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "nanochat/model/fp8/fp8_kernel.h"

namespace nanochat::kernels {

// logits: (rows, ld) bf16, columns [0, padded) used, padded % 8 == 0 and ld % 8 == 0. Writes loss[row] (if
// non-null) = lse - capped[target], 0 for target < 0. If grad, overwrites logits with d loss_row / d logits
// scaled by grad_scale[row * stride] (null = 1) and 1 / *num_valid (null = 1); columns >= vocab get 0. grad_amax
// (device float, may be null): set to the max |gradient| (as bf16), for its FP8 quantization. lse (may be null): each
// row's log-sum-exp of the capped logits, for softcap_ce_grad_mx (unset for target < 0).
void softcap_ce(
      void* logits, int64_t rows, int64_t ld, const int64_t* targets, int vocab, int padded, float softcap, float* loss,
      const float* grad_scale, int64_t grad_scale_stride, const int64_t* num_valid, bool grad, float* grad_amax,
      float* lse, cudaStream_t stream);

// softcap_ce's gradient (as its bf16), MX-quantized (quantize_mx) without writing it: from the logits (not
// overwritten) and softcap_ce's lse. rows % 32 == 0, padded % 64 == 0.
void softcap_ce_grad_mx(
      const void* logits, int64_t rows, int64_t ld, const int64_t* targets, const float* lse, int vocab, int padded,
      float softcap, const float* grad_scale, int64_t grad_scale_stride, const int64_t* num_valid, MxOut out,
      MxOut out_t, cudaStream_t stream);

} // namespace nanochat::kernels
