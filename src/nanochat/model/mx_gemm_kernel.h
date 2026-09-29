// MXFP8 GEMMs through CUTLASS (external/cutlass): sm_120a block-scaled mma.sync, ping-pong warp-specialized kernel,
// 128x128x128 tiles.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "nanochat/model/fp8_kernel.h"

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

// h (M, N) bf16 = a . b^T as cutlass_mx_gemm_bf16, and quantize_mx(h, relu_square)'s outputs (fp8_kernel.h), bit for
// bit, from the epilogue: q (M, N) e4m3 + q_scale, q_t (N, M) + q_t_scale. q_t_fp4.data set: q_t as NVFP4 instead.
const char* cutlass_mx_gemm_relu_square(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* h, void* q, void* q_scale,
      void* q_t, void* q_t_scale, const Nvfp4Out& q_t_fp4, int64_t M, int64_t N, int64_t K, cudaStream_t stream);

// ga = a . b^T (bf16, not written) and h (M, N) bf16 -> quantize_mx_relu_square_bwd's outputs from the epilogue, bit
// for bit: dh (M, N) e4m3 + dh_scale, dh_t (N, M) + dh_t_scale. dh_t_fp4.data set: dh_t as NVFP4 instead.
const char* cutlass_mx_gemm_relu_square_bwd(
      const void* a, const void* a_scale, const void* b, const void* b_scale, const void* h, void* dh, void* dh_scale,
      void* dh_t, void* dh_t_scale, const Nvfp4Out& dh_t_fp4, int64_t M, int64_t N, int64_t K, cudaStream_t stream);

// cutlass_mx_gemm_relu_square_bwd with NVFP4 a, b (nvfp4_gemm_kernel.h's layout; alpha: the device fp32 product of
// their tensor scales) and dh as NVFP4 (dh, required); dh_t as there. K % 256.
const char* cutlass_nvfp4_gemm_relu_square_bwd(
      const void* a, const void* a_scale, const void* b, const void* b_scale, const float* alpha, const void* h,
      const Nvfp4Out& dh, void* dh_t, void* dh_t_scale, const Nvfp4Out& dh_t_fp4, int64_t M, int64_t N, int64_t K,
      cudaStream_t stream);

} // namespace nanochat::kernels
