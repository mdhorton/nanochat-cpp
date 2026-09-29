// NVFP4 backward GEMMs: the operands in NVFP4, multiplied by CUTLASS's NVFP4 GEMM (nvfp4_gemm_kernel.h), with NVIDIA's
// recipe: for wgrad a 16-point random Hadamard transform along K (tokens), and stochastic rounding of the gradient
// (wgrad and dgrad). The kernels that quantize the operands write them as NVFP4 directly (Nvfp4Target); others' MXFP8
// operands are converted (mx_to_nvfp4, wgrad only).
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

struct Nvfp4Backward {
  bool wgrad = true, dgrad = false; // the Linears' weight / input gradient GEMMs in NVFP4
  bool rht = true, sr = true;       // wgrad: Hadamard on both operands; stochastic rounding of the gradient
  bool sr_dgrad = true; // dgrad: stochastic rounding of the gradient (no Hadamard; weights round to nearest)
  uint64_t seed = 0;    // stochastic rounding's
};

// Process-wide: the Linears' backward GEMMs (but lm_head's) in NVFP4 per options; null (default): MXFP8. options must
// outlive its use.
void set_nvfp4_backward(const Nvfp4Backward* options);
const Nvfp4Backward* nvfp4_backward();

// What a GEMM operand's NVFP4 copy is for: blocks along K, rounding per nvfp4_backward().
enum class Nvfp4Role {
  None,
  WgradInput,  // the input's transpose (K = tokens)
  WgradGrad,   // the output gradient's transpose (K = tokens)
  DgradGrad,   // the output gradient (K = out features)
  DgradWeight, // the weight's transpose (K = out features)
};

// A target for an operand (R, C), blocks along C, when role's GEMM is in NVFP4; else undefined data.
Nvfp4Target nvfp4_target(int64_t R, int64_t C, const torch::TensorOptions& options, Nvfp4Role role);

// bf16 (M, N) = a (M, K) . b (N, K)^T with the tensor scales applied (M, N % 128, K % 256)
torch::Tensor nvfp4_gemm(const Nvfp4Tensor& a, const Nvfp4Tensor& b);

// The GEMM's alpha as a device scalar: a's and b's tensor scales (times alpha when defined)
torch::Tensor nvfp4_alpha(const Nvfp4Tensor& a, const Nvfp4Tensor& b, const torch::Tensor& alpha = {});

// out (n, C) fp32 (+)= alpha * go_t (n, N) . in_t (C, N)^T in NVFP4. Each operand NVFP4 already (data, scale, amax), or
// MX (amax undefined) converted per nvfp4_backward() (then set).
void nvfp4_grad_weight(
      const torch::Tensor& go_t, const torch::Tensor& go_scale_t, const torch::Tensor& go_amax,
      const torch::Tensor& in_t, const torch::Tensor& in_scale_t, const torch::Tensor& in_amax,
      const torch::Tensor& out, bool accumulate, const torch::Tensor& alpha = {});

} // namespace nanochat
