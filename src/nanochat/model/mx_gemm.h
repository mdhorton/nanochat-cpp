// The MXFP8 GEMMs: cuBLASLt (_scaled_mm, and an fp32 output that can accumulate, which _scaled_mm lacks: weight
// gradients go straight into .grad) or CUTLASS (mx_gemm_kernel.h), per set_mx_gemm_backend.
#pragma once

#include <torch/torch.h>

#include "nanochat/model/fp8_kernel.h"
#include "nanochat/model/nvfp4.h"

namespace nanochat {

enum class MxGemmBackend { Cublas, Cutlass };

// process-wide; default Cublas
void set_mx_gemm_backend(MxGemmBackend backend);
MxGemmBackend mx_gemm_backend();

// The NVFP4 GEMMs (nvfp4.h's nvfp4_gemm, nvfp4_gemm_f32): CUTLASS (nvfp4_gemm_kernel.h) or cuBLASLt. Auto: per
// shape, whichever timed faster on its first call (cuBLASLt by >= 2%; its summation order differs, so Auto's bits
// can change between runs). Process-wide; default Cutlass (bench_gemm --fp4: cuBLASLt 3-13% slower on sm120).
enum class Nvfp4GemmBackend { Auto, Cutlass, Cublas };
void set_nvfp4_gemm_backend(Nvfp4GemmBackend backend);
Nvfp4GemmBackend nvfp4_gemm_backend();

// out (M, N) bf16 or fp32 (+)= alpha * a (M, K) . b (N, K)^T with the tensor scales applied (nvfp4_gemm_f32's
// contract; accumulate and alpha: fp32 out) through cuBLASLt: VEC16_UE4M3 block scales in CUTLASS's layout, the
// tensor scales as a device alpha
void nvfp4_gemm_cublas(
      const Nvfp4Tensor& a, const Nvfp4Tensor& b, const torch::Tensor& out, bool accumulate,
      const torch::Tensor& alpha = {});

// as nvfp4_gemm_cublas, per set_nvfp4_gemm_backend
void nvfp4_gemm_into(
      const Nvfp4Tensor& a, const Nvfp4Tensor& b, const torch::Tensor& out, bool accumulate,
      const torch::Tensor& alpha = {});

// a (M, K) . b (N, K)^T as out_dtype: a, b contiguous e4m3 with their MX scales, as _scaled_mm(a, b.t(), a_scale,
// b_scale). Cutlass: bf16 or fp32 out, dims % 128 (else _scaled_mm).
torch::Tensor mx_gemm(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      torch::ScalarType out_dtype);

// mx_gemm into out (contiguous, its dtype)
void mx_gemm_out(
      const torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
      const torch::Tensor& b_scale);

// h = mx_gemm(a, ..., bf16) into h and q = quantize_mx(h, true, true, true) into q, q_t (N, M) in one CUTLASS GEMM.
// q_t_fp4.data set: q_t as NVFP4 there instead (q_t, q_t_scale unused). False (nothing written): the Cublas backend,
// or dims not % 128.
bool mx_gemm_relu_square(
      const torch::Tensor& h, const torch::Tensor& q, const torch::Tensor& q_scale, const torch::Tensor& q_t,
      const torch::Tensor& q_t_scale, const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
      const torch::Tensor& b_scale, const kernels::Nvfp4Out& q_t_fp4 = {});

// mx_gemm_relu_square with NVFP4 a (M, K), b (N, K): q as NVFP4 (out.fp4, required), q_t (out_t) MX or NVFP4 (fp8.h's
// mx_outs views). False (nothing written): the Cublas backend.
bool nvfp4_gemm_relu_square(
      const Nvfp4Tensor& a, const Nvfp4Tensor& b, const torch::Tensor& h, const kernels::MxOut& out,
      const kernels::MxOut& out_t);

// relu^2's backward from its dgrad GEMM in one CUTLASS GEMM: ga = mx_gemm(a, ..., bf16) (not written) and h (M, N)
// bf16 -> quantize_mx_relu_square_bwd's outputs (fp8_kernel.h): dh rows and dh_t (N, M), each e4m3 with its MX scales.
// dh_t_fp4.data set: dh_t as NVFP4 there instead. False (nothing written): the Cublas backend, or dims not % 128.
bool mx_gemm_relu_square_bwd(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      const torch::Tensor& h, const torch::Tensor& dh, const torch::Tensor& dh_scale, const torch::Tensor& dh_t,
      const torch::Tensor& dh_t_scale, const kernels::Nvfp4Out& dh_t_fp4 = {});

// mx_gemm_relu_square_bwd with NVFP4 a (M, K), b (N, K): dh as NVFP4 (out.fp4, required), dh_t (out_t) MX or NVFP4
// (fp8.h's mx_outs views). False (nothing written): the Cublas backend.
bool nvfp4_gemm_relu_square_bwd(
      const Nvfp4Tensor& a, const Nvfp4Tensor& b, const torch::Tensor& h, const kernels::MxOut& out,
      const kernels::MxOut& out_t);

// out (M, N) fp32 (+)= alpha * a (M, K) . b (N, K)^T: a, b contiguous e4m3 with their MX scales (fp8.h's quantize_mx
// layout), dims % 128. accumulate: out += (beta 1), else out =. alpha: a device fp32 scalar (undefined: 1). Runs on
// the current stream with torch's cuBLASLt workspace. Cutlass: where cuBLASLt's heuristic splits K
// for the shape, whichever timed faster on its first call (CUTLASS by >= 2%).
void mx_gemm_f32(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      const torch::Tensor& out, bool accumulate, const torch::Tensor& alpha = {});

} // namespace nanochat
