// Fused smear (gpt.py: x = cat(x[:, :1], x[:, 1:] + smear_lambda * sigmoid(smear_gate(x[:, 1:, :24])) * x[:, :-1])),
// one kernel each way instead of torch's slices, broadcasts and their backward's fills, copies and adds.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// x (B, T, C) bf16 contiguous; w (1, gate_cols) and lambda (1) fp32. Forward rounds as the bf16 ops (the gate logit's
// 24-term sum may differ in order: rare 1-ulp differences); backward is fp32 inside, dw and dlambda deterministic.
torch::Tensor smear(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& lambda);

// Whether smear handles these dims: cols % 8 == 0 and <= 2048, gate_cols % 8 == 0 and <= 256.
bool smear_fits(int64_t cols, int64_t gate_cols);

} // namespace nanochat
