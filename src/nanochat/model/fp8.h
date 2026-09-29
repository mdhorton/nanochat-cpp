// FP8 training. Tensorwise (port of nanochat/fp8.py): the three matmuls of a Linear run through _scaled_mm, inputs
// and weights as e4m3, gradients as e5m2, one dynamic scale per tensor. Mx (not in Python): MXFP8, every operand e4m3
// with one power-of-two scale per 32 values along the contraction dim. Saves the fp8 input for backward.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <torch/torch.h>

#include "nanochat/model/fp8_kernel.h"
#include "nanochat/model/nvfp4.h"

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

struct Fp8Tensor;

// The MLP's c_proj(relu(c_fc(x)).square()) with both matmuls as fp8_matmul, bit for bit. x_2d (N, in) bf16 (N % 16
// == 0), w_fc (hidden, in), w_proj (out, hidden) -> (N, out). x_mx (Mx only, see relu_square_mlp_mx): x_2d already
// quantized (residual_norm_mx); x_2d then only carries autograd.
torch::Tensor fp8_relu_square_mlp(
      const torch::Tensor& x_2d, const torch::Tensor& w_fc, const torch::Tensor& w_proj,
      Fp8WeightCache* fc_cache = nullptr, Fp8WeightCache* proj_cache = nullptr,
      Fp8Recipe recipe = Fp8Recipe::Tensorwise, const Fp8Tensor* x_mx = nullptr);

// Whether fp8_relu_square_mlp of an (N, in) input runs under Mx (else tensorwise).
bool relu_square_mlp_mx(
      int64_t N, int64_t in, const torch::Tensor& w_fc, const torch::Tensor& w_proj, Fp8Recipe recipe);

// Attention's c_q, c_k, c_v as fp8_matmuls of the same x_2d (N, in) bf16, merged: x quantized once, one GEMM for the
// forward and one for the weight gradients (both bit-identical); the input gradient is summed in fp32 (Mx: one GEMM).
// Returns {q, k, v} (views of one (N, q + k + v) buffer), plus x_2d[:, :gate_cols] when gate_cols > 0 (the
// value-embedding gate's input, its gradient added in the same pass).
torch::autograd::variable_list fp8_qkv(
      const torch::Tensor& x_2d, const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv,
      int64_t gate_cols = 0, Fp8WeightCache* cache = nullptr, Fp8Recipe recipe = Fp8Recipe::Tensorwise);

// A quantized 2D tensor: data, its transpose (contiguous), and the scales that dequantize each (data * inv_scale).
// Tensorwise: one fp32 inverse scale for both. Mx: e8m0 block scales in cuBLAS's swizzled layout, inv_scale along
// data's rows, inv_scale_t along data_t's; for NVFP4 weight gradients (fp4_t()), data_t is NVFP4 instead (nvfp4.h's
// Nvfp4Tensor: data_t, inv_scale_t, amax_t), written by the producer as fp4_target and completed by finish_fp4_t.
struct Fp8Tensor {
  torch::Tensor data, data_t, inv_scale, inv_scale_t, amax_t;
  Nvfp4Target fp4_target;

  const torch::Tensor& inv_t() const {
    return inv_scale_t.defined() ? inv_scale_t : inv_scale;
  }

  bool fp4_t() const {
    return data_t.defined() && data_t.scalar_type() == torch::kUInt8;
  }
};

// A weight-gradient GEMM operand's role, for NVFP4 weight gradients: with nvfp4_wgrad set, Input and Grad transposes
// are NVFP4 (Grad: stochastic rounding).
enum class WgradRole { None, Input, Grad };

// Tensorwise. The fused kernels (fused = true) and the torch ops of fp8.py's _to_fp8 give identical bits.
Fp8Tensor quantize_fp8(const torch::Tensor& x, torch::ScalarType dtype, bool fused = true);

// MXFP8 e4m3 of a contiguous 2D bf16 or fp32 x, rows and cols % 128; rows / cols: write data / data_t.
// relu_square: of bf16(relu(x)^2). role: data_t as NVFP4 (see WgradRole).
Fp8Tensor quantize_mx(
      const torch::Tensor& x, bool rows = true, bool cols = true, bool relu_square = false,
      WgradRole role = WgradRole::None);

inline bool mx_fits(int64_t rows, int64_t cols) {
  return rows % 128 == 0 && cols % 128 == 0;
}

// Uninitialized MX buffers for a (R, C) tensor, dims % 128; rows / cols: allocate data / data_t with their scales.
// role: the transpose as an NVFP4 fp4_target instead (see WgradRole).
Fp8Tensor empty_mx(
      int64_t R, int64_t C, const torch::TensorOptions& options, bool rows = true, bool cols = true,
      WgradRole role = WgradRole::None);

// The kernels' views of q's buffers (null data where not allocated; out_t: fp4_target's when set).
std::pair<kernels::MxOut, kernels::MxOut> mx_outs(const Fp8Tensor& q);

// After q's producers: fp4_target (if any) completed into data_t, inv_scale_t, amax_t.
void finish_fp4_t(Fp8Tensor& q);

// e8m0 scales of a (rows, cols) MX tensor: cols / 32 blocks per row, swizzled (mx_fits: no padding).
torch::Tensor empty_mx_scale(int64_t rows, int64_t cols, const torch::TensorOptions& options);

// The kernels' view of MX data (R, ld) from (row, col) on, with its scales; row, col % 128 == 0.
kernels::MxOut mx_out(const torch::Tensor& data, const torch::Tensor& scale, int64_t row, int64_t col);

// quantize_mx of x into given buffers (e.g. mx_out views of a larger tensor).
void quantize_mx_into(const torch::Tensor& x, bool relu_square, kernels::MxOut out, kernels::MxOut out_t);

// Merged q/k/v weights under MX: {w_cat (n, C), w_cat^T, their scales}, from cache (optional) while unchanged.
std::vector<torch::Tensor> mx_qkv_weights(
      const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv, Fp8WeightCache* cache);

// A weight's e4m3 copy (tensorwise or Mx), from cache (optional) while w is unchanged.
Fp8Tensor quantize_fp8_weight(const torch::Tensor& w, Fp8WeightCache* cache, Fp8Recipe recipe = Fp8Recipe::Tensorwise);

// quantize_fp8 (fused) with max|x| already in scalars[0], e.g. from the kernel that wrote x; scalars: 2 device floats,
// scalars[1] gets the inverse scale. x: aligned, contiguous 2D bf16 or fp32.
Fp8Tensor quantize_fp8_amax_ready(const torch::Tensor& x, torch::ScalarType dtype, const torch::Tensor& scalars);

// Whether the MX backward may write w's gradient straight into w.grad (mx_gemm.h: fp32, += across micro-steps; no bf16
// copy for autograd to cast and add): a contiguous fp32 leaf that requires grad.
bool mx_grad_direct(const torch::Tensor& w);

// dW = alpha * go_t (n, N) . in_t (C, N)^T into the .grad of ws, whose rows concatenate to n: allocated (=) when
// undefined, else accumulated (+=). Several ws (merged q/k/v): their .grads are the row blocks of one buffer, so one
// GEMM serves. alpha: a device fp32 scalar (undefined: 1). Operands MX, or NVFP4 (uint8 data with its amax, see
// Fp8Tensor). nvfp4: in NVFP4 when set_nvfp4_wgrad is on (nvfp4.h), converting MX operands; NVFP4 ones always are.
void mx_grad_weights(
      const torch::Tensor& go_t, const torch::Tensor& go_scale_t, const torch::Tensor& in_t,
      const torch::Tensor& in_scale_t, const std::vector<torch::Tensor>& ws, const torch::Tensor& alpha = {},
      bool nvfp4 = true, const torch::Tensor& go_amax = {}, const torch::Tensor& in_amax = {});

// base_train.py's fp8_module_filter: dims divisible by 16 and both >= 128.
inline bool fp8_eligible(int64_t in_features, int64_t out_features) {
  return in_features % 16 == 0 && out_features % 16 == 0 && std::min(in_features, out_features) >= 128;
}

} // namespace nanochat
