// MXFP8 GEMMs through CUTLASS (external/cutlass): sm_120a block-scaled mma.sync, ping-pong warp-specialized kernel,
// 128x128x128 tiles.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// d (M, N) row-major = alpha * a (M, K) . b (N, K)^T: a, b row-major e4m3 with ue8m0 scales per 32 along K in cuBLAS's
// swizzled layout (fp8.h's quantize_mx), M, N, K % 128, pointers 16-byte aligned. Return an error message, or null.
const char* cutlass_mx_gemm_bf16(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      cudaStream_t stream);

// fp32 d; accumulate: d += (beta 1), else d =. alpha: a device fp32 scalar (null: 1).
const char* cutlass_mx_gemm_f32(
      const void* a, const void* a_scale, const void* b, const void* b_scale, float* d, int64_t M, int64_t N, int64_t K,
      const float* alpha, bool accumulate, cudaStream_t stream);

} // namespace nanochat::kernels
