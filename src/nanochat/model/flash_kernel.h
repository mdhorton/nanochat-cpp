// bf16 flash attention forward for sm_120 (mma.sync m16n8k16), causal with an optional left window.
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

} // namespace nanochat::kernels
