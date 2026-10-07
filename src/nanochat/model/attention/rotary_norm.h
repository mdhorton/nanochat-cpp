// Fused rotary embedding + QK norm: one CUDA kernel each way instead of ~10 elementwise ops per tensor.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// x: (B, T, H, D) bf16 q or k, contiguous; cos, sin: (1, >= T, 1, D / 2) bf16.
// Returns scale * rms_norm(apply_rotary_emb(x, cos, sin)) as gpt.py, but in fp32 with one rounding to bf16.
// Backward recomputes the rotation from x (x and one fp32 per row are saved).
torch::Tensor rotary_rms_norm(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin, double scale);

} // namespace nanochat
