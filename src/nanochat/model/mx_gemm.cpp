#include "nanochat/model/mx_gemm.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

#include "nanochat/model/mx_gemm_kernel.h"

namespace nanochat {

namespace {

void check(cublasStatus_t st, const char* what) {
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, what, ": ", cublasLtGetStatusName(st));
}

template <class T>
void set_attr(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, const T& v) {
  check(cublasLtMatmulDescSetAttribute(desc, attr, &v, sizeof(v)), "cublasLtMatmulDescSetAttribute");
}

struct Descriptors {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  ~Descriptors() {
    if (pref != nullptr)
      cublasLtMatmulPreferenceDestroy(pref);
    for (auto* l : {la, lb, lc})
      if (l != nullptr)
        cublasLtMatrixLayoutDestroy(l);
    if (desc != nullptr)
      cublasLtMatmulDescDestroy(desc);
  }
};

// the heuristic's choice per shape (it sees no pointers)
struct Algo {
  cublasLtMatmulAlgo_t algo;
  bool split_k;
  std::optional<bool> cutlass; // with split_k, under Cutlass: CUTLASS timed faster
};

std::mutex algo_mutex;
using AlgoKey = std::tuple<int64_t, int64_t, int64_t, bool, bool, size_t>;
std::map<AlgoKey, Algo> algos;

// CUTLASS over cuBLASLt's split-K only when this much faster: near ties keep cuBLASLt, so the choice rarely flips
constexpr float kCutlassMargin = 0.98f;

// median of 5 timed calls after 2 warm-ups, in us (synchronizes the stream)
template <class F>
float time_us(F&& f, cudaStream_t stream) {
  cudaEvent_t e0, e1;
  C10_CUDA_CHECK(cudaEventCreate(&e0));
  C10_CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<float> ts;
  for (int i = 0; i < 7; ++i) {
    C10_CUDA_CHECK(cudaEventRecord(e0, stream));
    f();
    C10_CUDA_CHECK(cudaEventRecord(e1, stream));
    C10_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    C10_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (i >= 2)
      ts.push_back(ms);
  }
  C10_CUDA_CHECK(cudaEventDestroy(e0));
  C10_CUDA_CHECK(cudaEventDestroy(e1));
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2] * 1e3f;
}

// device beta pointers, {0, 1} per device: alpha and beta share one pointer mode
const float* device_beta(bool one, const torch::Device& device) {
  static std::mutex mutex;
  static std::map<int, torch::Tensor> betas;
  const std::lock_guard lock(mutex);
  auto& t = betas[device.index()];
  if (!t.defined())
    t = torch::tensor({0.f, 1.f}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
  return t.data_ptr<float>() + (one ? 1 : 0);
}

std::atomic<MxGemmBackend> backend{MxGemmBackend::Cublas};

void check_cutlass(const char* error) {
  TORCH_CHECK(error == nullptr, error);
}

bool cutlass_fits(const torch::Tensor& a, const torch::Tensor& b, torch::ScalarType out_dtype) {
  return backend == MxGemmBackend::Cutlass && (out_dtype == torch::kBFloat16 || out_dtype == torch::kFloat32) &&
         a.is_contiguous() && b.is_contiguous() && a.size(0) % 128 == 0 && b.size(0) % 128 == 0 && a.size(1) % 128 == 0;
}

void cutlass_out(
      const torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
      const torch::Tensor& b_scale) {
  TORCH_CHECK(
        out.is_contiguous() && out.size(0) == a.size(0) && out.size(1) == b.size(0) && b.size(1) == a.size(1),
        "mx_gemm: shape mismatch");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t M = a.size(0), N = b.size(0), K = a.size(1);
  if (out.scalar_type() == torch::kBFloat16)
    check_cutlass(
          kernels::cutlass_mx_gemm_bf16(
                a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(), out.data_ptr(), M, N, K, stream));
  else
    check_cutlass(
          kernels::cutlass_mx_gemm_f32(
                a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(), out.data_ptr<float>(), M, N, K,
                nullptr, false, stream));
}

void check_mx_out(const torch::Tensor& t, const torch::Tensor& scale, int64_t rows, int64_t cols, const char* what) {
  TORCH_CHECK(
        t.is_contiguous() && t.scalar_type() == torch::kFloat8_e4m3fn && t.size(0) == rows && t.size(1) == cols &&
              scale.numel() == rows * cols / 32,
        what, ": expected e4m3 (", rows, ", ", cols, ") with its MX scales");
}

} // namespace

void set_mx_gemm_backend(MxGemmBackend b) {
  backend = b;
}

MxGemmBackend mx_gemm_backend() {
  return backend;
}

torch::Tensor mx_gemm(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      torch::ScalarType out_dtype) {
  if (!cutlass_fits(a, b, out_dtype))
    return at::_scaled_mm(a, b.t(), a_scale, b_scale, {}, {}, out_dtype, false);
  auto out = torch::empty({a.size(0), b.size(0)}, a.options().dtype(out_dtype));
  cutlass_out(out, a, a_scale, b, b_scale);
  return out;
}

void mx_gemm_out(
      const torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
      const torch::Tensor& b_scale) {
  if (!cutlass_fits(a, b, out.scalar_type())) {
    auto o = out;
    at::_scaled_mm_out(o, a, b.t(), a_scale, b_scale, {}, {}, out.scalar_type(), false);
    return;
  }
  cutlass_out(out, a, a_scale, b, b_scale);
}

bool mx_gemm_relu_square(
      const torch::Tensor& h, const torch::Tensor& q, const torch::Tensor& q_scale, const torch::Tensor& q_t,
      const torch::Tensor& q_t_scale, const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
      const torch::Tensor& b_scale, const kernels::Nvfp4Out& q_t_fp4) {
  if (!cutlass_fits(a, b, h.scalar_type()) || h.scalar_type() != torch::kBFloat16)
    return false;
  const int64_t M = a.size(0), N = b.size(0), K = a.size(1);
  TORCH_CHECK(
        h.is_contiguous() && h.size(0) == M && h.size(1) == N && b.size(1) == K, "mx_gemm_relu_square: shape mismatch");
  check_mx_out(q, q_scale, M, N, "mx_gemm_relu_square");
  const bool fp4 = q_t_fp4.data != nullptr;
  if (!fp4)
    check_mx_out(q_t, q_t_scale, N, M, "mx_gemm_relu_square");
  check_cutlass(
        kernels::cutlass_mx_gemm_relu_square(
              a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(), h.data_ptr(), q.data_ptr(),
              q_scale.data_ptr(), fp4 ? nullptr : q_t.data_ptr(), fp4 ? nullptr : q_t_scale.data_ptr(), q_t_fp4, M, N,
              K, at::cuda::getCurrentCUDAStream().stream()));
  return true;
}

bool mx_gemm_relu_square_bwd(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      const torch::Tensor& h, const torch::Tensor& dh, const torch::Tensor& dh_scale, const torch::Tensor& dh_t,
      const torch::Tensor& dh_t_scale, const kernels::Nvfp4Out& dh_t_fp4) {
  if (!cutlass_fits(a, b, h.scalar_type()) || h.scalar_type() != torch::kBFloat16)
    return false;
  const int64_t M = a.size(0), N = b.size(0), K = a.size(1);
  TORCH_CHECK(
        h.is_contiguous() && h.size(0) == M && h.size(1) == N && b.size(1) == K,
        "mx_gemm_relu_square_bwd: shape mismatch");
  check_mx_out(dh, dh_scale, M, N, "mx_gemm_relu_square_bwd");
  const bool fp4 = dh_t_fp4.data != nullptr;
  if (!fp4)
    check_mx_out(dh_t, dh_t_scale, N, M, "mx_gemm_relu_square_bwd");
  check_cutlass(
        kernels::cutlass_mx_gemm_relu_square_bwd(
              a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(), h.data_ptr(), dh.data_ptr(),
              dh_scale.data_ptr(), fp4 ? nullptr : dh_t.data_ptr(), fp4 ? nullptr : dh_t_scale.data_ptr(), dh_t_fp4, M,
              N, K, at::cuda::getCurrentCUDAStream().stream()));
  return true;
}

bool nvfp4_gemm_relu_square_bwd(
      const Nvfp4Tensor& a, const Nvfp4Tensor& b, const torch::Tensor& h, const kernels::MxOut& out,
      const kernels::MxOut& out_t) {
  if (backend != MxGemmBackend::Cutlass || h.scalar_type() != torch::kBFloat16)
    return false;
  const int64_t M = a.data.size(0), N = b.data.size(0), K = a.data.size(1) * 2;
  TORCH_CHECK(
        h.is_contiguous() && h.size(0) == M && h.size(1) == N && b.data.size(1) * 2 == K && out.fp4.data != nullptr &&
              out.fp4.ld == N && (out_t.fp4.data != nullptr ? out_t.fp4.ld == M : out_t.ld == M),
        "nvfp4_gemm_relu_square_bwd: shape mismatch");
  const auto alpha = nvfp4_alpha(a, b);
  check_cutlass(
        kernels::cutlass_nvfp4_gemm_relu_square_bwd(
              a.data.data_ptr(), a.scale.data_ptr(), b.data.data_ptr(), b.scale.data_ptr(), alpha.data_ptr<float>(),
              h.data_ptr(), out.fp4, out_t.data, out_t.scale, out_t.fp4, M, N, K,
              at::cuda::getCurrentCUDAStream().stream()));
  return true;
}

void mx_gemm_f32(
      const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b, const torch::Tensor& b_scale,
      const torch::Tensor& out, bool accumulate, const torch::Tensor& alpha) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2, "mx_gemm_f32: expected 2D tensors");
  const int64_t M = a.size(0), K = a.size(1), N = b.size(0);
  TORCH_CHECK(
        b.size(1) == K && out.size(0) == M && out.size(1) == N && M % 128 == 0 && N % 128 == 0 && K % 128 == 0,
        "mx_gemm_f32: shape mismatch or dims not % 128");
  TORCH_CHECK(
        a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn &&
              out.scalar_type() == torch::kFloat32 && a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
        "mx_gemm_f32: expected contiguous e4m3 operands and an fp32 output");
  TORCH_CHECK(
        a_scale.scalar_type() == torch::kFloat8_e8m0fnu && b_scale.scalar_type() == torch::kFloat8_e8m0fnu &&
              a_scale.numel() == M * K / 32 && b_scale.numel() == N * K / 32,
        "mx_gemm_f32: expected e8m0 MX scales");
  const bool device_scalars = alpha.defined();
  TORCH_CHECK(
        !device_scalars || (alpha.numel() == 1 && alpha.scalar_type() == torch::kFloat32 && alpha.is_cuda()),
        "mx_gemm_f32: alpha must be a device fp32 scalar");
  const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  const auto stream = at::cuda::getCurrentCUDAStream();
  void* workspace = at::cuda::getCUDABlasLtWorkspace();
  const size_t ws = at::cuda::getCUDABlasLtWorkspaceSize();

  // cuBLASLt is column-major: D^T (N, M) = b (K-major, op T) . a^T (K-major, op N), C = D = out
  Descriptors d;
  check(cublasLtMatmulDescCreate(&d.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
  set_attr(d.desc, CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
  set_attr(d.desc, CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
  if (device_scalars)
    set_attr<int32_t>(d.desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_POINTER_MODE_DEVICE);
  set_attr<const void*>(d.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, b_scale.data_ptr());
  set_attr<const void*>(d.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, a_scale.data_ptr());
  set_attr(d.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
  set_attr(d.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
  check(cublasLtMatrixLayoutCreate(&d.la, CUDA_R_8F_E4M3, K, N, K), "cublasLtMatrixLayoutCreate");
  check(cublasLtMatrixLayoutCreate(&d.lb, CUDA_R_8F_E4M3, K, M, K), "cublasLtMatrixLayoutCreate");
  check(cublasLtMatrixLayoutCreate(&d.lc, CUDA_R_32F, N, M, N), "cublasLtMatrixLayoutCreate");

  Algo algo;
  const AlgoKey key{M, N, K, accumulate, device_scalars, ws};
  {
    const std::lock_guard lock(algo_mutex);
    auto it = algos.find(key);
    if (it == algos.end()) {
      check(cublasLtMatmulPreferenceCreate(&d.pref), "cublasLtMatmulPreferenceCreate");
      check(cublasLtMatmulPreferenceSetAttribute(d.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)),
            "cublasLtMatmulPreferenceSetAttribute");
      cublasLtMatmulHeuristicResult_t heur{};
      int n = 0;
      check(cublasLtMatmulAlgoGetHeuristic(handle, d.desc, d.la, d.lb, d.lc, d.lc, d.pref, 1, &heur, &n),
            "cublasLtMatmulAlgoGetHeuristic");
      TORCH_CHECK(
            n > 0 && heur.state == CUBLAS_STATUS_SUCCESS, "mx_gemm_f32: no cuBLASLt algorithm for ", M, "x", N, "x", K);
      int32_t split_k = 1;
      size_t written = 0;
      check(cublasLtMatmulAlgoConfigGetAttribute(
                  &heur.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k), &written),
            "cublasLtMatmulAlgoConfigGetAttribute");
      it = algos.emplace(key, Algo{heur.algo, split_k != 1, {}}).first;
    }
    algo = it->second;
  }
  const auto run_cutlass = [&](const torch::Tensor& o) {
    check_cutlass(
          kernels::cutlass_mx_gemm_f32(
                a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(), o.data_ptr<float>(), M, N, K,
                device_scalars ? alpha.data_ptr<float>() : nullptr, accumulate, stream.stream()));
  };
  const float one = 1.f, host_beta = accumulate ? 1.f : 0.f;
  const float* alpha_ptr = device_scalars ? alpha.data_ptr<float>() : &one;
  const float* beta_ptr = device_scalars ? device_beta(accumulate, out.device()) : &host_beta;
  const auto run_cublas = [&](const torch::Tensor& o) {
    check(cublasLtMatmul(
                handle, d.desc, alpha_ptr, b.data_ptr(), d.la, a.data_ptr(), d.lb, beta_ptr, o.data_ptr(), d.lc,
                o.data_ptr(), d.lc, &algo.algo, workspace, ws, stream),
          "cublasLtMatmul");
  };
  // CUTLASS has no split-K, which cuBLASLt picks for few output tiles over a long K (weight gradients): there, time
  // both once (into scratch) and keep the faster. cuBLASLt without split-K is bit-identical to CUTLASS.
  bool cutlass = backend == MxGemmBackend::Cutlass;
  if (cutlass && algo.split_k) {
    if (!algo.cutlass) {
      const auto scratch = torch::zeros_like(out);
      const float t_cutlass = time_us(
            [&] {
              run_cutlass(scratch);
            },
            stream.stream());
      const float t_cublas = time_us(
            [&] {
              run_cublas(scratch);
            },
            stream.stream());
      algo.cutlass = t_cutlass < kCutlassMargin * t_cublas;
      const std::lock_guard lock(algo_mutex);
      algos[key].cutlass = algo.cutlass;
    }
    cutlass = *algo.cutlass;
  }
  if (cutlass)
    run_cutlass(out);
  else
    run_cublas(out);
}

} // namespace nanochat
