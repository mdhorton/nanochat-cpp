// NVFP4 quantization for CUTLASS's NVFP4 GEMMs (nvfp4_gemm_kernel.h), from MXFP8 operands (no torch headers, so nvcc
// stays out of libtorch). Rounding as nvfp4.cuh, so the results match nvfp4_fake_quant's bit for bit.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "nanochat/model/fp8_kernel.h"

namespace nanochat::kernels {

// MX (rows, cols) row-major e4m3 with e8m0 scales per 32 along cols in the swizzled layout (fp8.h's quantize_mx) ->
// NVFP4 (rows, cols): out two e2m1 per byte (the even column in the low nibble), out_scale ue4m3 per 16 along cols in
// the same swizzled layout (nvfp4_scale_bytes), under the tensor scale *amax / (6 * 448). hadamard (optional): a 16x16
// row-major fp32 matrix that multiplies each 16 values first (x . H). amax: a zeroed device float, set to the max
// magnitude (after the transform). stochastic: stochastic rounding from seed. rows % 128, cols % 128.
void mx_to_nvfp4(
      const void* data, const void* scale, int64_t rows, int64_t cols, const float* hadamard, bool stochastic,
      uint64_t seed, void* out, void* out_scale, float* amax, cudaStream_t stream);

// x (rows, cols) row-major fp32 (else bf16) -> NVFP4 in 16x16 blocks (one scale each, round to nearest) as producers
// write it (fp8_kernel.h's Nvfp4Out): out along cols, and out_t (data null: none) x's transpose along rows, the same
// values (W and W^T for forward and dgrad). rows, cols % 16.
void quantize_nvfp4_2d(
      const void* x, bool x_f32, int64_t rows, int64_t cols, const Nvfp4Out& out, const Nvfp4Out& out_t,
      cudaStream_t stream);

// Completes NVFP4 (rows, cols) written by producers (fp8_kernel.h's Nvfp4Out): the tensor scale 2^-k with the largest
// k keeping every block scale within 448, so out_scale = ue4m3(scale16 * 2^k) (exact but for e4m3's subnormals), and
// *amax = 6 * 448 * 2^-k for that tensor scale. scale16: rows * cols / 16 bf16 in out_scale's swizzled layout (as
// mx_to_nvfp4's); smax: kNvfp4Slots, their max's bits (those of epoch). rows % 128, cols % 64.
struct Nvfp4Finish {
  const void* scale16;
  const unsigned long long* smax;
  uint32_t epoch;
  int64_t rows, cols;
  void* out_scale;
  float* amax;
};

// a, and b too when given, in one launch
void nvfp4_finish(const Nvfp4Finish& a, const Nvfp4Finish* b, cudaStream_t stream);

// NVFP4 as mx_to_nvfp4 writes it -> bf16 (rows, cols)
void nvfp4_to_bf16(
      const void* data, const void* scale, const float* amax, int64_t rows, int64_t cols, void* out,
      cudaStream_t stream);

// MX as mx_to_nvfp4 reads it -> bf16 (rows, cols), exact
void mx_to_bf16(const void* data, const void* scale, int64_t rows, int64_t cols, void* out, cudaStream_t stream);

} // namespace nanochat::kernels
