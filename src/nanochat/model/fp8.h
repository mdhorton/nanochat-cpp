// FP8 training with tensorwise dynamic scaling. Port of nanochat/fp8.py: the three matmuls of a Linear run through
// _scaled_mm, inputs and weights as e4m3, gradients as e5m2, one scale per tensor. Saves the fp8 input for backward.
#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace nanochat {

// input_2d (N, in) bf16, weight (out, in) any float dtype -> (N, out) in input_2d's dtype.
torch::Tensor fp8_matmul(const torch::Tensor& input_2d, const torch::Tensor& weight);

// The MLP's c_proj(relu(c_fc(x)).square()) with both matmuls as fp8_matmul, bit for bit. x_2d (N, in) bf16 (N % 16
// == 0), w_fc (hidden, in), w_proj (out, hidden) -> (N, out).
torch::Tensor fp8_relu_square_mlp(const torch::Tensor& x_2d, const torch::Tensor& w_fc, const torch::Tensor& w_proj);

// Attention's c_q, c_k, c_v as fp8_matmuls of the same x_2d (N, in) bf16, merged: x quantized once, one GEMM for the
// forward and one for the weight gradients (both bit-identical); the input gradient is summed in fp32. Returns {q, k,
// v} (views of one (N, q + k + v) buffer), plus x_2d[:, :gate_cols] when gate_cols > 0 (the value-embedding gate's
// input, its gradient added in the same pass).
torch::autograd::variable_list fp8_qkv(
      const torch::Tensor& x_2d, const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv,
      int64_t gate_cols = 0);

// Tensorwise quantization of a 2D tensor: data, its transpose (contiguous), inverse scale. The fused kernels
// (fused = true) and the torch ops of fp8.py's _to_fp8 give identical bits.
struct Fp8Tensor {
  torch::Tensor data, data_t, inv_scale;
};

Fp8Tensor quantize_fp8(const torch::Tensor& x, torch::ScalarType dtype, bool fused = true);

// base_train.py's fp8_module_filter: dims divisible by 16 and both >= 128.
inline bool fp8_eligible(int64_t in_features, int64_t out_features) {
  return in_features % 16 == 0 && out_features % 16 == 0 && std::min(in_features, out_features) >= 128;
}

} // namespace nanochat
