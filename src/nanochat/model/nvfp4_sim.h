// Simulated NVFP4 training for a Linear: operands fake-quantized to NVFP4 (nvfp4_sim_kernel.h) and multiplied in bf16
// with fp32 accumulation, which gives the NVFP4 GEMM's result. Per NVIDIA's NVFP4 pretraining recipe: 16-value e4m3
// block scales plus an fp32 tensor scale, 16x16 weight blocks (forward and dgrad see the same weight), a random
// Hadamard transform on the weight gradient's operands, stochastic rounding of gradients; each also selectable per
// GEMM. GEMMs not simulated run as MXFP8 (fp8.h). Slow: for judging numerics, not speed.
#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace nanochat {

struct Nvfp4Options {
  bool fwd = true, dgrad = true, wgrad = true;               // which GEMMs are NVFP4; the others MXFP8
  bool rht_fwd = false, rht_dgrad = false, rht_wgrad = true; // 16-point random Hadamard transform along K
  bool sr_dgrad = true, sr_wgrad = true;                     // stochastic rounding of the gradient operand
  bool weight_2d = true;                                     // 16x16 weight blocks (without rht)
  uint64_t seed = 0;                                         // stochastic rounding's
};

// x (R, C) bf16 or fp32 -> bf16 NVFP4 values, blocks of 16 along C (C % 16). blocks_2d: 16x16 blocks (R % 16).
// stochastic: fresh random bits per call, from seed.
torch::Tensor nvfp4_fake_quant(
      const torch::Tensor& x, bool stochastic = false, bool blocks_2d = false, uint64_t seed = 0);

// x (R, C) with each 16 columns multiplied by the random Hadamard matrix (orthonormal, fixed signs); fp32 result.
torch::Tensor nvfp4_rht(const torch::Tensor& x);

// input_2d (N, in) bf16, weight (out, in) fp32 -> (N, out) bf16. Dims % 128 (MXFP8 GEMMs) and % 16.
torch::Tensor nvfp4_sim_matmul(const torch::Tensor& input_2d, const torch::Tensor& weight, const Nvfp4Options& options);

} // namespace nanochat
