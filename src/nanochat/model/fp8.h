// FP8 training. Tensorwise (port of nanochat/fp8.py): the three matmuls of a Linear run through _scaled_mm, inputs
// and weights as e4m3, gradients as e5m2, one dynamic scale per tensor. Mx (not in Python): MXFP8, every operand e4m3
// with one power-of-two scale per 32 values along the contraction dim. Saves the fp8 input for backward.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <torch/torch.h>

namespace nanochat {

enum class Fp8Recipe { Tensorwise, Mx };

// FP8 copies of weights, remade only when a weight changes: in-place updates (optimizer, load) bump its version, so
// the micro-steps between optimizer steps quantize it once, with the same bits. Disabled: remade on every call.
class Fp8WeightCache {
public:
  bool enabled = false;
  // make()'s tensors, reused while kind is the same and every weight keeps its storage and version
  std::vector<torch::Tensor> get(
        int64_t kind, const std::vector<torch::Tensor>& weights,
        const std::function<std::vector<torch::Tensor>()>& make);

private:
  struct Key {
    const void* impl;
    const void* data;
    int64_t version;
  };

  int64_t kind_ = -1;
  std::vector<Key> keys_;
  std::vector<torch::Tensor> value_;
};

// input_2d (N, in) bf16, weight (out, in) any float dtype -> (N, out) in input_2d's dtype.
// cache (optional): the weight's FP8 copy. Mx needs N, in, out % 128 (else tensorwise).
torch::Tensor fp8_matmul(
      const torch::Tensor& input_2d, const torch::Tensor& weight, Fp8WeightCache* cache = nullptr,
      Fp8Recipe recipe = Fp8Recipe::Tensorwise);

// The MLP's c_proj(relu(c_fc(x)).square()) with both matmuls as fp8_matmul, bit for bit. x_2d (N, in) bf16 (N % 16
// == 0), w_fc (hidden, in), w_proj (out, hidden) -> (N, out).
torch::Tensor fp8_relu_square_mlp(
      const torch::Tensor& x_2d, const torch::Tensor& w_fc, const torch::Tensor& w_proj,
      Fp8WeightCache* fc_cache = nullptr, Fp8WeightCache* proj_cache = nullptr,
      Fp8Recipe recipe = Fp8Recipe::Tensorwise);

// Attention's c_q, c_k, c_v as fp8_matmuls of the same x_2d (N, in) bf16, merged: x quantized once, one GEMM for the
// forward and one for the weight gradients (both bit-identical); the input gradient is summed in fp32 (Mx: one GEMM).
// Returns {q, k, v} (views of one (N, q + k + v) buffer), plus x_2d[:, :gate_cols] when gate_cols > 0 (the
// value-embedding gate's input, its gradient added in the same pass).
torch::autograd::variable_list fp8_qkv(
      const torch::Tensor& x_2d, const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv,
      int64_t gate_cols = 0, Fp8WeightCache* cache = nullptr, Fp8Recipe recipe = Fp8Recipe::Tensorwise);

// A quantized 2D tensor: data, its transpose (contiguous), and the scales that dequantize each (data * inv_scale).
// Tensorwise: one fp32 inverse scale for both. Mx: e8m0 block scales in cuBLAS's swizzled layout, inv_scale along
// data's rows, inv_scale_t along data_t's.
struct Fp8Tensor {
  torch::Tensor data, data_t, inv_scale, inv_scale_t;

  const torch::Tensor& inv_t() const {
    return inv_scale_t.defined() ? inv_scale_t : inv_scale;
  }
};

// Tensorwise. The fused kernels (fused = true) and the torch ops of fp8.py's _to_fp8 give identical bits.
Fp8Tensor quantize_fp8(const torch::Tensor& x, torch::ScalarType dtype, bool fused = true);

// MXFP8 e4m3 of a contiguous 2D bf16 or fp32 x, rows and cols % 128; rows / cols: write data / data_t.
// relu_square: of bf16(relu(x)^2).
Fp8Tensor quantize_mx(const torch::Tensor& x, bool rows = true, bool cols = true, bool relu_square = false);

inline bool mx_fits(int64_t rows, int64_t cols) {
  return rows % 128 == 0 && cols % 128 == 0;
}

// A weight's e4m3 copy (tensorwise or Mx), from cache (optional) while w is unchanged.
Fp8Tensor quantize_fp8_weight(const torch::Tensor& w, Fp8WeightCache* cache, Fp8Recipe recipe = Fp8Recipe::Tensorwise);

// quantize_fp8 (fused) with max|x| already in scalars[0], e.g. from the kernel that wrote x; scalars: 2 device floats,
// scalars[1] gets the inverse scale. x: aligned, contiguous 2D bf16 or fp32.
Fp8Tensor quantize_fp8_amax_ready(const torch::Tensor& x, torch::ScalarType dtype, const torch::Tensor& scalars);

// base_train.py's fp8_module_filter: dims divisible by 16 and both >= 128.
inline bool fp8_eligible(int64_t in_features, int64_t out_features) {
  return in_features % 16 == 0 && out_features % 16 == 0 && std::min(in_features, out_features) >= 128;
}

} // namespace nanochat
