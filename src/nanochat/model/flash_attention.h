// FlashAttention-2 (external/flash-attention, hdim 128 bf16 kernels) built for this GPU, as
// at::_flash_attention_forward with is_causal and PyTorch's backward.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// q, k, v (B, T, H, D) bf16, last dim contiguous, D == 128, H the same for all. Causal; window >= 0: `window` keys to
// the left plus the query. Softmax scale 1 / sqrt(D).
torch::Tensor flash_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window);

// Whether flash_attention handles these inputs.
bool flash_attention_fits(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);

} // namespace nanochat
