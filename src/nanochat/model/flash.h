// Causal attention with a bf16 flash-attention forward for sm_120 (flash_kernel.cu) and FA2's backward.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// q: (B, T, H, 128), k, v: (B, T, Hkv, 128) bf16, heads contiguous within a token (views of a merged qkv are fine).
// window < 0: full causal, else keys t - window .. t (as fa2_attention in gpt.cpp). T % 128 == 0, H % Hkv == 0.
// Returns (B, T, H, 128) bf16. Backward is at::_flash_attention_backward with this forward's output and log-sum-exp.
torch::Tensor flash_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window);

// Whether flash_attention takes these inputs (the checks above).
bool flash_supported(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);

// The forward alone (out, lse); variant: kernels::flash_fwd's tile configuration (tests, benchmarks).
std::pair<torch::Tensor, torch::Tensor> flash_forward(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window, int variant = -1);

} // namespace nanochat
