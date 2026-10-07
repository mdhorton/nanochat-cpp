// flash attention for sm_120 (mma.sync), causal with an optional left window: bf16 forward, bf16 and MXFP8 backward.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

inline constexpr int kFlashHeadDim = 128;
inline constexpr int kFlashSeqMultiple = 128; // seq_len must divide by this
inline constexpr int kFlashVariants = 5;      // tile configurations, for tuning (variant < 0: the default)

// q: (B, T, H, 128), k, v: (B, T, Hkv, 128) bf16, heads contiguous within a token, tokens q_ld / k_ld / v_ld elements
// apart. window < 0: full causal, else keys t - window .. t; softmax scale 1/sqrt(128). H % Hkv == 0.
// out: (B, T, H, 128) bf16, contiguous; lse: (B, H, T) fp32, natural log of the softmax denominator (FA2's).
void flash_fwd(
      const void* q, const void* k, const void* v, void* out, float* lse, int B, int64_t T, int H, int Hkv,
      int64_t q_ld, int64_t k_ld, int64_t v_ld, int64_t window, cudaStream_t stream, int variant = -1);

// "warps x rows per warp, key tile" of a variant
const char* flash_variant_name(int variant);

inline constexpr int kFlashBwdDqVariants = 2, kFlashBwdDkvVariants = 4; // tile configurations (< 0: the default)

// flash_fwd's backward from dout (B, T, H, 128) bf16, contiguous, and the forward's inputs, out (contiguous) and lse.
// dq: (B, T, H, 128), dk, dv: (B, T, Hkv, 128) bf16, contiguous. delta: (B, H, T) fp32 scratch.
void flash_bwd(
      const void* dout, const void* q, const void* k, const void* v, const void* out, const float* lse, float* delta,
      void* dq, void* dk, void* dv, int B, int64_t T, int H, int Hkv, int64_t q_ld, int64_t k_ld, int64_t v_ld,
      int64_t window, cudaStream_t stream, int dq_variant = -1, int dkv_variant = -1);

const char* flash_bwd_dq_variant_name(int variant);
const char* flash_bwd_dkv_variant_name(int variant);

// MXFP8 backward (mx_flash_bwd_kernel.cu, sm_120a): as flash_bwd with every matmul block-scaled e4m3 (ue8m0 per 32).
// Inputs quantized twice, along head_dim (rows, for S = Q Kᵀ and dP = dout Vᵀ) and along tokens (transposed, for
// dv = Pᵀ dout, dk = dSᵀ Q, dq = dS K).

// x: (B, T, heads, 128) bf16, tokens x_ld elements apart. data: (B, T, heads, 128) e4m3; scale: (B, heads, T) u32,
// byte j = ue8m0 exponent of dims [32j, 32j + 32). With out ((B, T, heads, 128) bf16, contiguous): also
// delta = rowsum(x * out), (B, heads, T) fp32.
void mx_flash_quantize_rows(
      const void* x, int64_t x_ld, void* data, uint32_t* scale, const void* out, float* delta, int B, int64_t T,
      int heads, cudaStream_t stream);

// x as above. data: (B, heads, 128, T) e4m3, x transposed with tokens permuted in each 16 as
// [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] (the order an accumulator lands in as an A fragment).
// scale: (B, heads, T / 32, 128) ue8m0 per (32 tokens, dim), dim d at byte d % 8 * 16 + d / 16 * 2 + d / 8 % 2 (one
// 16-byte load per mma thread; mx_flash.cuh t_scale_pos).
void mx_flash_quantize_t(
      const void* x, int64_t x_ld, void* data, uint8_t* scale, int B, int64_t T, int heads, cudaStream_t stream);

// dout and out (B, T, heads, 128) bf16, contiguous, read once: dout as mx_flash_quantize_rows (data, scale, and
// delta) and as mx_flash_quantize_t (data_t, scale_t). T % 64 == 0.
void mx_flash_quantize_dout(
      const void* dout, const void* out, void* data, uint32_t* scale, void* data_t, uint8_t* scale_t, float* delta,
      int B, int64_t T, int heads, cudaStream_t stream);

struct MxFlashBwdInputs {
  const void *q, *k, *v, *dout;                             // mx_flash_quantize_rows
  const uint32_t *q_scale, *k_scale, *v_scale, *dout_scale; // (B, heads, T)
  const void *qt, *kt, *doutt;                              // mx_flash_quantize_t
  const uint8_t *qt_scale, *kt_scale, *doutt_scale;         // (B, heads, T / 32, 128)
  const float *lse, *delta;                                 // (B, H, T)
};

inline constexpr int kMxFlashBwdDqVariants = 2, kMxFlashBwdDkvVariants = 4; // tile configurations (< 0: the default)

// dq: (B, T, H, 128), dk, dv: (B, T, Hkv, 128) bf16, contiguous.
void mx_flash_bwd(
      const MxFlashBwdInputs& in, void* dq, void* dk, void* dv, int B, int64_t T, int H, int Hkv, int64_t window,
      cudaStream_t stream, int dq_variant = -1, int dkv_variant = -1);

const char* mx_flash_bwd_dq_variant_name(int variant);
const char* mx_flash_bwd_dkv_variant_name(int variant);

} // namespace nanochat::kernels
