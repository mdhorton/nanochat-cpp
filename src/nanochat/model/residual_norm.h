// Fused residual step: residual add, per-layer lambda blend and rms_norm in one CUDA kernel each way. Backward also
// folds in autograd's accumulation of the residual's two gradients (the norm's and the stream's).
#pragma once

#include <torch/torch.h>

namespace nanochat {

// Per row of the last dim: s = r.defined() ? x + r : x; res = x0.defined() ? resid_lambdas[layer] * s +
// x0_lambdas[layer] * x0 : s. Returns {res, rms_norm(res)}. The add and blend round as gpt.py's bf16 ops (res and,
// without a norm gradient, all gradients are bit-identical); the norm is fp32 inside. Tensors: same-shape contiguous
// bf16, last dim % 8 == 0 and <= 2048; lambdas (n_layer) fp32, their gradients summed in fp32 (deterministic).
std::pair<torch::Tensor, torch::Tensor> residual_norm(
      const torch::Tensor& x, const torch::Tensor& r, const torch::Tensor& x0 = {},
      const torch::Tensor& resid_lambdas = {}, const torch::Tensor& x0_lambdas = {}, int64_t layer = 0);

} // namespace nanochat
