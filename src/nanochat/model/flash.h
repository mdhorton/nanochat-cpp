// Causal attention with bf16 flash-attention kernels for sm_120 (flash_kernel.cu, flash_bwd_kernel.cu).
#pragma once

#include <torch/torch.h>

namespace nanochat {

// q: (B, T, H, 128), k, v: (B, T, Hkv, 128) bf16, heads contiguous within a token (views of a merged qkv are fine).
// window < 0: full causal, else keys t - window .. t (as fa2_attention in gpt.cpp). T % 128 == 0, H % Hkv == 0.
// Returns (B, T, H, 128) bf16. Backward: FA2's (at::_flash_attention_backward) on this forward's out and lse.
torch::Tensor flash_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window);

// Whether flash_attention takes these inputs (the checks above).
bool flash_supported(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);

// The forward alone (out, lse); variant: kernels::flash_fwd's tile configuration (tests, benchmarks).
std::pair<torch::Tensor, torch::Tensor> flash_forward(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window, int variant = -1);

// The backward alone: dq, dk, dv from dout and the forward's inputs, out and lse; variants: kernels::flash_bwd's.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_backward(
      const torch::Tensor& dout, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
      const torch::Tensor& out, const torch::Tensor& lse, int64_t window, int dq_variant = -1, int dkv_variant = -1);

} // namespace nanochat
