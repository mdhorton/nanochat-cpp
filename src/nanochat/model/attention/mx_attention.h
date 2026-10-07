// Attention's q, k, v under MXFP8 in one autograd function: the merged q/k/v GEMM (fp8_qkv), rotary + QK norm
// (rotary_rms_norm) and the value-embedding mix. Backward writes dq, dk (from the rotary norm's backward) and dv
// (from the mix's) straight into the GEMMs' MX buffers, never in bf16.
#pragma once

#include <torch/torch.h>

#include "nanochat/model/fp8/fp8.h"

namespace nanochat {

// x_2d (N, C) bf16; wq (H * D, C), wk, wv (Hkv * D, C); cos, sin (1, >= T, 1, D / 2) bf16, T = N's sequence
// length. ve (N, Hkv * D) bf16 or undefined; with it, w_gate (Hkv, gate_cols). Returns (N, *) bf16:
//   q, k = rotary_rms_norm(x @ wq^T, x @ wk^T, cos, sin, scale), as (N, H * D) and (N, Hkv * D)
//   v = x @ wv^T + 3 sigmoid(x[:, :gate_cols] @ w_gate^T) * ve per head (ve given), else x @ wv^T
// Bit-identical to fp8_qkv (Mx), the ops and rotary_rms_norm, except the gate logits' gradient (its sum's order).
// x_mx: x_2d already quantized (residual_norm_mx); x_2d then carries autograd and the gate's columns only.
// quantize_attention: also returns mx_flash_attention's inputs (MxFlashInputs' 12 tensors in order; no gradient),
// written by the rotary norm and the value mix. q, k and (with ve) v are then unwritten placeholders that only carry
// autograd (mx_flash_attention with them as pre reads nothing else). T % 128 == 0.
torch::autograd::variable_list mx_attention_inputs(
      const torch::Tensor& x_2d, const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv,
      const torch::Tensor& cos, const torch::Tensor& sin, double scale, const torch::Tensor& ve,
      const torch::Tensor& w_gate, int64_t head_dim, Fp8WeightCache* cache = nullptr, const Fp8Tensor* x_mx = nullptr,
      bool quantize_attention = false);

// Whether mx_attention_inputs handles these shapes: MX dims (% 128), head_dim 128.
bool mx_attention_fits(int64_t N, int64_t C, int64_t n_q, int64_t n_kv, int64_t head_dim);

} // namespace nanochat
