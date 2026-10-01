// Simulated NVFP4 training for a Linear: operands fake-quantized to NVFP4 (nvfp4_sim_kernel.h) and multiplied in bf16
// with fp32 accumulation, which gives the NVFP4 GEMM's result. Per NVIDIA's NVFP4 pretraining recipe: 16-value e4m3
// block scales plus an fp32 tensor scale, 16x16 weight blocks (forward and dgrad see the same weight), a random
// Hadamard transform on the weight gradient's operands, stochastic rounding of gradients; each also selectable per
// GEMM, as is Quartet II's MS-EDEN for the backward GEMMs. GEMMs not simulated run as MXFP8 (fp8.h). Slow: for judging
// numerics, not speed.
#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace nanochat {

struct Nvfp4Options {
  bool fwd = true, dgrad = true, wgrad = true;               // which GEMMs are NVFP4; the others MXFP8
  bool rht_fwd = false, rht_dgrad = false, rht_wgrad = true; // 16-point random Hadamard transform along K
  bool sr_dgrad = true, sr_wgrad = true;                     // stochastic rounding of the gradient operand
  bool eden_dgrad = false, eden_wgrad = false;               // MS-EDEN (nvfp4_eden) on both operands, not rht / sr
  int64_t eden_group = 128;                                  // MS-EDEN rotation and correction group (power of 2)
  bool eden_fixed_signs = false;                             // MS-EDEN signs per (step, Linear, GEMM), not per call
  bool weight_2d = true;                                     // 16x16 weight blocks (without rht)
  uint64_t seed = 0;                                         // stochastic rounding's
  int64_t step = 0;                                          // the optimizer step (eden_fixed_signs)
};

// x (R, C) bf16 or fp32 -> bf16 NVFP4 values, blocks of 16 along C (C % 16). blocks_2d: 16x16 blocks (R % 16).
// stochastic: fresh random bits per call, from seed.
torch::Tensor nvfp4_fake_quant(
      const torch::Tensor& x, bool stochastic = false, bool blocks_2d = false, uint64_t seed = 0);

// x (R, C) with each 16 columns multiplied by the random Hadamard matrix (orthonormal, fixed signs); fp32 result.
torch::Tensor nvfp4_rht(const torch::Tensor& x);

// MS-EDEN (Quartet II, arXiv 2601.22813) for a backward GEMM's operands. Both are rotated along K by one rotation
// (nvfp4_eden_rotation, G x G); values round to nearest; each 16-block's e4m3 scale is multiplied by its G-group's
// ||x||² / <x, q> and rounded stochastically, so that E<x, dequantized> = ||x||².
struct Nvfp4Eden {
  torch::Tensor values; // (R, C) bf16: FP4 values times the corrected block scales (exact)
  torch::Tensor scale;  // fp32 scalar tensor scale: x . h ~ values * scale
};

// x (R, C) bf16 or fp32 with C % G, h (G, G) from nvfp4_eden_rotation. Fresh random bits per call, from seed.
Nvfp4Eden nvfp4_eden(const torch::Tensor& x, const torch::Tensor& h, uint64_t seed = 0);

// diag(random signs) . G x G Hadamard / sqrt(G) (G a power of 2, >= 16): orthonormal, fp32. fresh: new signs per
// call (from seed); else the signs are seed's.
torch::Tensor nvfp4_eden_rotation(
      const torch::Device& device, uint64_t seed = 0, int64_t group = 128, bool fresh = true);

// input_2d (N, in) bf16, weight (out, in) fp32 -> (N, out) bf16. Dims % 128 (MXFP8 GEMMs) and % 16. id: the Linear's
// (eden_fixed_signs); eden false: this Linear's backward GEMMs use rht / sr instead of MS-EDEN.
torch::Tensor nvfp4_sim_matmul(
      const torch::Tensor& input_2d, const torch::Tensor& weight, const Nvfp4Options& options, int64_t id = 0,
      bool eden = true);

} // namespace nanochat
