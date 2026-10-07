// Causal attention with MXFP8 flash-attention kernels (sm_120a): mx_flash_kernel.cu's forward, mx_flash_bwd_kernel.cu's
// backward, both on one quantization of q, k, v.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// q, k, v MX-quantized (layouts as kernels::mx_flash_quantize_rows / _t): along head_dim (q, k, v: (B, T, heads, 128)
// e4m3 as uint8, scales (B, heads, T) int32 words) and along tokens (qt, kt, vt: (B, heads, 128, T) permuted, scales
// (B, heads, T / 32, 128) uint8). The forward reads q, k, vt; the backward all but vt.
struct MxFlashInputs {
  torch::Tensor q, q_scale, k, k_scale, v, v_scale;
  torch::Tensor qt, qt_scale, kt, kt_scale, vt, vt_scale;
};

// q: (B, T, H, 128), k, v: (B, T, Hkv, 128) bf16, heads contiguous within a token (views of a merged qkv are fine).
// window < 0: full causal, else keys t - window .. t (as fa2_attention in gpt.cpp). T % 128 == 0, H % Hkv == 0.
// Returns (B, T, H, 128) bf16. pre: q, k, v already quantized (mx_attention_inputs), else quantized here. Saves the
// quantized inputs, not q, k, v.
torch::Tensor mx_flash_attention(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
      const MxFlashInputs* pre = nullptr);

// Pieces, exposed for tests, benchmarks and flash_attention's MX backward.
// x (B, T, heads, 128) bf16 along head_dim (data, scale) and along tokens (data, scale).
std::pair<torch::Tensor, torch::Tensor> mx_flash_quantize_rows(const torch::Tensor& x);
std::pair<torch::Tensor, torch::Tensor> mx_flash_quantize_t(const torch::Tensor& x);
MxFlashInputs mx_flash_quantize(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);
// the forward alone (out, lse); without pre, quantizes q, k and v (along tokens) itself
std::pair<torch::Tensor, torch::Tensor> mx_flash_forward(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
      const MxFlashInputs* pre = nullptr);
// dq, dk, dv from dout (B, T, H, 128) bf16, in (all but vt), the forward's out and lse; variants:
// kernels::mx_flash_bwd's
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mx_flash_backward(
      const torch::Tensor& dout, const MxFlashInputs& in, const torch::Tensor& out, const torch::Tensor& lse,
      int64_t window, int dq_variant = -1, int dkv_variant = -1);

} // namespace nanochat
