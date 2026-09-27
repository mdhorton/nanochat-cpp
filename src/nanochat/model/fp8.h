// FP8 training with tensorwise dynamic scaling. Port of nanochat/fp8.py: the three matmuls of a Linear run through
// _scaled_mm, inputs and weights as e4m3, gradients as e5m2, one scale per tensor. Saves the fp8 input for backward.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <torch/torch.h>

namespace nanochat {

// FP8 copies of weights, remade only when a weight changes: in-place updates (optimizer, load) bump its version, so
// the micro-steps between optimizer steps quantize it once, with the same bits. Disabled: remade on every call.
class Fp8WeightCache {
public:
  bool enabled = false;
  // make()'s tensors, reused while every weight keeps its storage and version
  std::vector<torch::Tensor> get(
        const std::vector<torch::Tensor>& weights, const std::function<std::vector<torch::Tensor>()>& make);

private:
  struct Key {
    const void* impl;
    const void* data;
    int64_t version;
  };

  std::vector<Key> keys_;
  std::vector<torch::Tensor> value_;
};

// input_2d (N, in) bf16, weight (out, in) any float dtype -> (N, out) in input_2d's dtype.
// cache (optional): the weight's FP8 copy.
torch::Tensor fp8_matmul(const torch::Tensor& input_2d, const torch::Tensor& weight, Fp8WeightCache* cache = nullptr);

// The MLP's c_proj(relu(c_fc(x)).square()) with both matmuls as fp8_matmul, bit for bit. x_2d (N, in) bf16 (N % 16
// == 0), w_fc (hidden, in), w_proj (out, hidden) -> (N, out).
torch::Tensor fp8_relu_square_mlp(
      const torch::Tensor& x_2d, const torch::Tensor& w_fc, const torch::Tensor& w_proj,
      Fp8WeightCache* fc_cache = nullptr, Fp8WeightCache* proj_cache = nullptr);

// Attention's c_q, c_k, c_v as fp8_matmuls of the same x_2d (N, in) bf16, merged: x quantized once, one GEMM for the
// forward and one for the weight gradients (both bit-identical); the input gradient is summed in fp32. Returns {q, k,
// v} (views of one (N, q + k + v) buffer), plus x_2d[:, :gate_cols] when gate_cols > 0 (the value-embedding gate's
// input, its gradient added in the same pass).
torch::autograd::variable_list fp8_qkv(
      const torch::Tensor& x_2d, const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv,
      int64_t gate_cols = 0, Fp8WeightCache* cache = nullptr);

// Tensorwise quantization of a 2D tensor: data, its transpose (contiguous), inverse scale. The fused kernels
// (fused = true) and the torch ops of fp8.py's _to_fp8 give identical bits.
struct Fp8Tensor {
  torch::Tensor data, data_t, inv_scale;
};

Fp8Tensor quantize_fp8(const torch::Tensor& x, torch::ScalarType dtype, bool fused = true);

// quantize_fp8(w, e4m3), from cache (optional) while w is unchanged.
Fp8Tensor quantize_fp8_weight(const torch::Tensor& w, Fp8WeightCache* cache);

// quantize_fp8 (fused) with max|x| already in scalars[0], e.g. from the kernel that wrote x; scalars: 2 device floats,
// scalars[1] gets the inverse scale. x: aligned, contiguous 2D bf16 or fp32.
Fp8Tensor quantize_fp8_amax_ready(const torch::Tensor& x, torch::ScalarType dtype, const torch::Tensor& scalars);

// base_train.py's fp8_module_filter: dims divisible by 16 and both >= 128.
inline bool fp8_eligible(int64_t in_features, int64_t out_features) {
  return in_features % 16 == 0 && out_features % 16 == 0 && std::min(in_features, out_features) >= 128;
}

} // namespace nanochat
