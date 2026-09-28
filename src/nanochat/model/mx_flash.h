// Causal attention with an MXFP8 flash-attention forward (sm_120a, mx_flash_kernel.cu) and FA2's bf16 backward.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// The forward's MX inputs (layouts as mx_flash_quantize_rows / mx_flash_quantize_vt), when a producer already wrote
// them (mx_attention_inputs).
struct MxFlashInputs {
  torch::Tensor q, q_scale, k, k_scale, vt, v_scale;
};

// q: (B, T, H, 128), k, v: (B, T, Hkv, 128) bf16, heads contiguous within a token (views of a merged qkv are fine).
// window < 0: full causal, else keys t - window .. t (as fa2_attention in gpt.cpp). T % 128 == 0, H % Hkv == 0.
// Returns (B, T, H, 128) bf16. Backward is at::_flash_attention_backward on the bf16 inputs with this forward's
// output and log-sum-exp. pre: q, k, v already quantized (else quantized here).
torch::Tensor mx_flash_attention(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
      const MxFlashInputs* pre = nullptr);

// Forward pieces, exposed for tests: MX quantization of q or k rows ((B, T, heads, 128) e4m3 as uint8, (B, heads, T)
// int32 scale words) and of Vᵀ ((B, heads, 128, T) permuted e4m3 as uint8, (B, heads, T / 64, 128, 2) uint8 scales),
// and the attention itself (returns out, lse).
std::pair<torch::Tensor, torch::Tensor> mx_flash_quantize_rows(const torch::Tensor& x);
std::pair<torch::Tensor, torch::Tensor> mx_flash_quantize_vt(const torch::Tensor& v);
std::pair<torch::Tensor, torch::Tensor> mx_flash_forward(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
      const MxFlashInputs* pre = nullptr);

} // namespace nanochat
