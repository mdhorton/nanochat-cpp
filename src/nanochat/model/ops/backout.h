// Fused backout (gpt.py: x = x - backout_lambda * x_backout before the final norm): one kernel each way instead of
// torch's mul, sub and their backward's neg, muls and reduction.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// x, xb: same-shape contiguous bf16; lambda: (1) fp32. Bit-identical to the ops but for dlambda (fp32 sum,
// deterministic). x's gradient is the incoming one, passed through.
torch::Tensor backout(const torch::Tensor& x, const torch::Tensor& xb, const torch::Tensor& lambda);

} // namespace nanochat
