// MXFP8 flash attention forward for sm_120a: block-scaled mma.sync (e4m3, ue8m0 per 32) for Q·Kᵀ and P·V.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

inline constexpr int kMxFlashHeadDim = 128;
inline constexpr int kMxFlashBlockM = 128; // seq_len must divide by this

// x: (B, T, heads, 128) bf16, tokens x_stride elements apart (heads contiguous within a token).
// data: (B, T, heads, 128) e4m3; scale: (B, heads, T) u32, byte j = ue8m0 exponent of dims [32j, 32j + 32).
void mx_flash_quantize_rows(
      const void* x, void* data, uint32_t* scale, int B, int64_t T, int heads, int64_t x_stride, cudaStream_t stream);

// v: (B, T, heads, 128) bf16, tokens v_stride elements apart. vt: (B, heads, 128, T) e4m3, i.e. V transposed, with
// tokens permuted in each 16 as [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] (the order P's accumulator lands in).
// scale: (B, heads, T / 64, 128, 2) ue8m0 per (dim, 32 tokens).
void mx_flash_quantize_vt(
      const void* v, void* vt, uint8_t* scale, int B, int64_t T, int heads, int64_t v_stride, cudaStream_t stream);

// Causal attention with an optional left window (window < 0: none; else keys t - window .. t), softmax scale
// 1/sqrt(128). q, k, vt and scales as above (H query heads, H % Hkv == 0). out: (B, T, H, 128) bf16, contiguous;
// lse: (B, H, T) fp32, natural log of the softmax denominator.
void mx_flash_fwd(
      const void* q, const uint32_t* q_scale, const void* k, const uint32_t* k_scale, const void* vt,
      const uint8_t* v_scale, void* out, float* lse, int B, int64_t T, int H, int Hkv, int64_t window,
      cudaStream_t stream);

} // namespace nanochat::kernels
