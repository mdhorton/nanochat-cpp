// Fused residual step: residual add, per-layer lambda blend and rms_norm in one CUDA kernel each way. Backward also
// folds in autograd's accumulation of the residual's two gradients (the norm's and the stream's).
#pragma once

#include <torch/torch.h>

#include "nanochat/model/fp8.h"

namespace nanochat {

// Shared by the residual_norm calls blending one x0: they sum x0's gradient in one buffer, in autograd's order
// (bit-identical, without autograd's adds). The first call returns the sum (and folds in its x's gradient if x is
// x0), so the later calls must depend on its outputs, as in a residual stream.
struct X0Grad : torch::CustomClassHolder {
  torch::Tensor sum;
  bool claimed = false;
};

// Per row of the last dim: s = r.defined() ? x + r : x; res = x0.defined() ? resid_lambdas[layer] * s +
// x0_lambdas[layer] * x0 : s. Returns {res, rms_norm(res)}. The add and blend round as gpt.py's bf16 ops (res and,
// without a norm gradient, all gradients are bit-identical); the norm is fp32 inside. Tensors: same-shape contiguous
// bf16, last dim % 8 == 0 and <= 2048; lambdas (n_layer) fp32, their gradients summed in fp32 (deterministic).
std::pair<torch::Tensor, torch::Tensor> residual_norm(
      const torch::Tensor& x, const torch::Tensor& r, const torch::Tensor& x0 = {},
      const torch::Tensor& resid_lambdas = {}, const torch::Tensor& x0_lambdas = {}, int64_t layer = 0,
      const c10::intrusive_ptr<X0Grad>& x0_grad = {});

// residual_norm for an MX consumer: the norm's output is written as its MXFP8 quantization (quantize_mx of n, bit
// for bit, both layouts) instead of in bf16. n is the autograd handle to pass along with n_mx (fp8_relu_square_mlp,
// mx_attention_inputs); only its first gate_cols columns hold values (the value-embedding gate's input).
struct ResidualNormMx {
  torch::Tensor res, n;
  Fp8Tensor n_mx;
};

ResidualNormMx residual_norm_mx(
      const torch::Tensor& x, const torch::Tensor& r, const torch::Tensor& x0 = {},
      const torch::Tensor& resid_lambdas = {}, const torch::Tensor& x0_lambdas = {}, int64_t layer = 0,
      const c10::intrusive_ptr<X0Grad>& x0_grad = {}, int64_t gate_cols = 0);

// Whether residual_norm_mx handles (rows, cols): MX dims (% 128), cols within the kernel's shared memory.
bool residual_norm_mx_fits(int64_t rows, int64_t cols);

} // namespace nanochat
