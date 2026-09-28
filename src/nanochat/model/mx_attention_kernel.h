// Kernels of mx_attention.h's q/k/v path (no torch headers, so nvcc stays out of libtorch). Token rows of
// heads * head_dim bf16 values; head_dim == kMxHeadDim.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "nanochat/model/fp8_kernel.h"

namespace nanochat::kernels {

inline constexpr int kMxHeadDim = 128;

// rotary_norm_bwd's dx (as its bf16), MX-quantized (quantize_mx) into out / out_t without writing it. dout contiguous;
// x's tokens x_stride apart; tokens % 32 == 0.
void rotary_norm_bwd_mx(
      const void* dout, const void* x, const void* cos, const void* sin, const float* rstd, int64_t tokens, int heads,
      int64_t seq_len, int64_t x_stride, float scale, MxOut out, MxOut out_t, cudaStream_t stream);

// out = v + 3 sigmoid(z) * ve per head, rounded as gpt.py's bf16 ops. v's tokens v_stride apart; ve, out contiguous;
// z: (tokens, heads).
void value_mix_fwd(
      const void* v, int64_t v_stride, const void* z, const void* ve, void* out, int64_t tokens, int heads,
      cudaStream_t stream);

// v (with ve: value_mix_fwd's out, also written to out) as mx_flash_fwd's Vᵀ, in kernels::mx_flash_quantize_vt's
// layouts. v's tokens v_stride apart, B * seq_len of them; seq_len % 64 == 0. Without ve, z and out are unused.
void value_mix_vt(
      const void* v, int64_t v_stride, const void* z, const void* ve, void* out, void* vt, uint8_t* vt_scale, int B,
      int64_t seq_len, int heads, cudaStream_t stream);

// value_mix_fwd's backward from dout (contiguous): dve = gate * dout, dz (the gate logits' gradient, its sum over
// head_dim in fp32), and v's gradient dout MX-quantized into out / out_t. tokens % 32 == 0.
void value_mix_bwd_mx(
      const void* dout, const void* z, const void* ve, void* dve, void* dz, int64_t tokens, int heads, MxOut out,
      MxOut out_t, cudaStream_t stream);

} // namespace nanochat::kernels
