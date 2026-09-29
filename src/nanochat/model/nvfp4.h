// NVFP4 weight gradients: a weight-gradient GEMM's operands in NVFP4, multiplied by CUTLASS's NVFP4 GEMM
// (nvfp4_gemm_kernel.h), with NVIDIA's recipe for wgrad: a 16-point random Hadamard transform along K (tokens) and
// stochastic rounding of the gradient. The kernels that quantize the operands write them as NVFP4 directly
// (Nvfp4Target); others' MXFP8 operands are converted (mx_to_nvfp4).
#pragma once

#include <cstdint>

#include <torch/torch.h>

#include "nanochat/model/fp8_kernel.h"

namespace nanochat {

// NVFP4 (R, C): data (R, C / 2) uint8, two e2m1 each; scale (R * C / 16) ue4m3 in the swizzled layout; amax: device
// fp32 scalar, the tensor scale's max|x| (x = e2m1 * scale * amax / (6 * 448))
struct Nvfp4Tensor {
  torch::Tensor data, scale, amax;
};

// MX data (R, C) e4m3 with its scales (fp8.h's quantize_mx layout) -> NVFP4, blocks of 16 along C (R, C % 128). rht:
// each 16 values times nvfp4_hadamard first. stochastic: stochastic rounding from seed.
Nvfp4Tensor mx_to_nvfp4(
      const torch::Tensor& data, const torch::Tensor& scale, bool rht, bool stochastic, uint64_t seed = 0);

// dequantized, bf16 (R, C)
torch::Tensor nvfp4_to_bf16(const Nvfp4Tensor& t);
torch::Tensor mx_to_bf16(const torch::Tensor& data, const torch::Tensor& scale);

// The random Hadamard matrix (16x16 fp32, orthonormal: Walsh-Hadamard / 4 with rows of fixed random signs), per device.
const torch::Tensor& nvfp4_hadamard(const torch::Device& device);

// Its rows' signs: bit i set when row i is negated
uint32_t nvfp4_hadamard_signs();

// NVFP4 (R, C) as a producer writes it (kernels::Nvfp4Out): data (R, C / 2), scale16 (R * C / 16) bf16 block scales
// without the tensor scale (swizzled as the final ones), smax (zeroed) their max; nvfp4_finish completes it. R % 128,
// C % 64.
struct Nvfp4Target {
  torch::Tensor data, scale16, smax;
  bool rht = false, stochastic = false;
  uint64_t seed = 0;
};

Nvfp4Target empty_nvfp4(
      int64_t R, int64_t C, const torch::TensorOptions& options, bool rht, bool stochastic, uint64_t seed = 0);

// The kernels' view of t from (row, col) on; row % 128, col % 64.
kernels::Nvfp4Out nvfp4_out(const Nvfp4Target& t, int64_t row = 0, int64_t col = 0);

// After its producers: the tensor scale and ue4m3 block scales (nvfp4_kernel.h's nvfp4_finish)
Nvfp4Tensor nvfp4_finish(const Nvfp4Target& t);

// out (M, N) fp32 (+)= alpha * a (M, K) . b (N, K)^T with the tensor scales applied. accumulate: +=. alpha: a device
// fp32 scalar (undefined: 1). M, N % 128, K % 256.
void nvfp4_gemm_f32(
      const Nvfp4Tensor& a, const Nvfp4Tensor& b, const torch::Tensor& out, bool accumulate,
      const torch::Tensor& alpha = {});

struct Nvfp4Wgrad {
  bool rht = true, sr = true; // Hadamard on both operands; stochastic rounding of the gradient
  uint64_t seed = 0;          // stochastic rounding's
};

// Process-wide: the Linears' weight gradients (mx_grad_weights, fp8.h) in NVFP4 per options; null (default): MXFP8.
// options must outlive its use.
void set_nvfp4_wgrad(const Nvfp4Wgrad* options);
const Nvfp4Wgrad* nvfp4_wgrad();

// With nvfp4_wgrad set: a target for a weight-gradient operand (R, C), K = C along tokens, per its options; grad: the
// output gradient (stochastic rounding), else the input. Else undefined data.
Nvfp4Target nvfp4_wgrad_target(int64_t R, int64_t C, const torch::TensorOptions& options, bool grad);

// out (n, C) fp32 (+)= alpha * go_t (n, N) . in_t (C, N)^T in NVFP4. Each operand NVFP4 already (data, scale, amax), or
// MX (amax undefined) converted per nvfp4_wgrad() (then set).
void nvfp4_grad_weight(
      const torch::Tensor& go_t, const torch::Tensor& go_scale_t, const torch::Tensor& go_amax,
      const torch::Tensor& in_t, const torch::Tensor& in_scale_t, const torch::Tensor& in_amax,
      const torch::Tensor& out, bool accumulate, const torch::Tensor& alpha = {});

} // namespace nanochat
