// Fused per-layer residual blend: one CUDA kernel forward, one (plus a tiny finalize) backward, instead of 3 and ~8.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// x, x0: same-shape bf16; resid_lambdas, x0_lambdas: (n_layer) fp32 parameters.
// Returns resid_lambdas[layer] * x + x0_lambdas[layer] * x0, bit-identical to gpt.py's ops (same bf16 roundings).
// The lambdas' gradients are summed in fp32 (deterministic).
torch::Tensor lambda_blend(
      const torch::Tensor& x, const torch::Tensor& x0, const torch::Tensor& resid_lambdas,
      const torch::Tensor& x0_lambdas, int64_t layer);

} // namespace nanochat
