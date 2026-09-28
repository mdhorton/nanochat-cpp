// cuBLASLt MXFP8 GEMM with an fp32 output that can accumulate (_scaled_mm has no beta): weight gradients go straight
// into .grad.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// out (M, N) fp32 (+)= alpha * a (M, K) . b (N, K)^T: a, b contiguous e4m3 with their MX scales (fp8.h's quantize_mx
// layout), dims % 128. accumulate: out += (beta 1), else out =. alpha: a device fp32 scalar (undefined: 1). Runs on
// the current stream with torch's cuBLASLt workspace.
void mx_gemm_f32(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      const torch::Tensor& out, bool accumulate, const torch::Tensor& alpha = {});

} // namespace nanochat
