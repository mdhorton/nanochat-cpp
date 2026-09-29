// NVFP4 GEMMs through CUTLASS (external/cutlass): sm_120a block-scaled mma.sync (kind::mxf4nvf4). For benchmarking
// against the MXFP8 ones (mx_gemm_kernel.h), and for NVFP4 backward GEMMs (nvfp4.h).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// the kernel configs (tile and schedule): count and names
int nvfp4_gemm_configs();
const char* nvfp4_gemm_config_name(int config);

// bytes of a (rows, K) operand's scales in CUTLASS's block-scaled layout
int64_t nvfp4_scale_bytes(int64_t rows, int64_t K);

// d (M, N) row-major, bf16 or (f32) fp32 = a (M, K) . b (N, K)^T: a, b row-major e2m1, two per byte (the even element
// in the low nibble), with ue4m3 scales per 16 along K (nvfp4_scale_bytes). M, N % 128, K % 256, pointers 16-byte
// aligned. Return an error message, or null.
const char* cutlass_nvfp4_gemm(
      int config, const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, bool f32, int64_t M,
      int64_t N, int64_t K, cudaStream_t stream);

// d (M, N) fp32 (+)= alpha * a . b^T as cutlass_nvfp4_gemm (128x128x128 pingpong). accumulate: d += (beta 1), else d =.
// alpha: a device fp32 scalar (null: 1), e.g. the product of the operands' tensor scales.
const char* cutlass_nvfp4_gemm_f32(
      const void* a, const void* a_scale, const void* b, const void* b_scale, float* d, int64_t M, int64_t N, int64_t K,
      const float* alpha, bool accumulate, cudaStream_t stream);

// d (M, N) bf16 = alpha * a . b^T as cutlass_nvfp4_gemm (128x128x128 pingpong); alpha: a device fp32 scalar
const char* cutlass_nvfp4_gemm_bf16(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      const float* alpha, cudaStream_t stream);

} // namespace nanochat::kernels
