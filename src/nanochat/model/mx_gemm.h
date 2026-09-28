// The MXFP8 GEMMs: cuBLASLt (_scaled_mm, and an fp32 output that can accumulate, which _scaled_mm lacks: weight
// gradients go straight into .grad) or CUTLASS (mx_gemm_kernel.h), per set_mx_gemm_backend.
#pragma once

#include <torch/torch.h>

namespace nanochat {

enum class MxGemmBackend { Cublas, Cutlass };

// process-wide; default Cublas
void set_mx_gemm_backend(MxGemmBackend backend);
MxGemmBackend mx_gemm_backend();

// a (M, K) . b (N, K)^T as out_dtype: a, b contiguous e4m3 with their MX scales, as _scaled_mm(a, b.t(), a_scale,
// b_scale). Cutlass: bf16 or fp32 out, dims % 128 (else _scaled_mm).
torch::Tensor mx_gemm(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      torch::ScalarType out_dtype);

// mx_gemm into out (contiguous, its dtype)
void mx_gemm_out(
      const torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
      const torch::Tensor& b_scale);

// out (M, N) fp32 (+)= alpha * a (M, K) . b (N, K)^T: a, b contiguous e4m3 with their MX scales (fp8.h's quantize_mx
// layout), dims % 128. accumulate: out += (beta 1), else out =. alpha: a device fp32 scalar (undefined: 1). Runs on
// the current stream with torch's cuBLASLt workspace. Cutlass: unless cuBLASLt's heuristic splits K for the shape.
void mx_gemm_f32(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      const torch::Tensor& out, bool accumulate, const torch::Tensor& alpha = {});

} // namespace nanochat
