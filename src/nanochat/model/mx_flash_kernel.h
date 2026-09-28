// MXFP8 flash attention forward for sm_120a: block-scaled mma.sync (e4m3, ue8m0 per 32) for Q·Kᵀ and P·V.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

inline constexpr int kMxFlashHeadDim = 128;
inline constexpr int kMxFlashBlockM = 128; // seq_len must divide by this

// Causal attention with an optional left window (window < 0: none; else keys t - window .. t), softmax scale
// 1/sqrt(128). q, k along head_dim and vt along tokens, as flash_kernel.h's flash_mx_quantize_rows / _t (H query
// heads, H % Hkv == 0). out: (B, T, H, 128) bf16, contiguous; lse: (B, H, T) fp32, natural log of the softmax
// denominator.
void mx_flash_fwd(
      const void* q, const uint32_t* q_scale, const void* k, const uint32_t* k_scale, const void* vt,
      const uint8_t* v_scale, void* out, float* lse, int B, int64_t T, int H, int Hkv, int64_t window,
      cudaStream_t stream);

} // namespace nanochat::kernels
