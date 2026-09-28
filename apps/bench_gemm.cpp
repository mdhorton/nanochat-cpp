// Times cuBLASLt's algorithms for the MXFP8 GEMM shapes of a training micro-step against at::_scaled_mm's choice.
// Usage: bench_gemm [--tokens T] [--embd C] [--algos N] [--f32] [--shape M,N,K] [--torch-ws-mb MB]
// Also reports whether each algorithm gives the same bits twice (split-K with in-place reduction does not).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <torch/torch.h>

#include "nanochat/model/fp8.h"

using nanochat::quantize_mx;

#define LT_CHECK(call)                                                                                                 \
  do {                                                                                                                 \
    const cublasStatus_t st_ = (call);                                                                                 \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                                                                \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cublasLtGetStatusName(st_));                \
      std::exit(1);                                                                                                    \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

namespace {

// row-major D (M, N) = A (M, K) . B (N, K)^T, both operands K-major e4m3 with MX scales
struct Shape {
  int64_t M, N, K;
  bool f32_out;
  std::string what;
};

struct Timing {
  double ms; // per call
  double tflops;
};

// median of `iters` timed calls after `warmup`
template <class F>
Timing time_calls(F&& f, int64_t flops, int warmup = 5, int iters = 20) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  for (int i = 0; i < warmup; ++i)
    f();
  std::vector<float> ms(iters);
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(t0, stream);
    f();
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms[i], t0, t1);
  }
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  std::sort(ms.begin(), ms.end());
  const double med = ms[iters / 2];
  return {med, static_cast<double>(flops) / med / 1e9};
}

struct AlgoInfo {
  int id = -1;
  uint32_t tile = 0, stages = 0, cluster = 0, swizzle = 0, custom = 0, reduction = 0;
  int32_t splitk = 1;
  uint32_t inner = 0;
};

AlgoInfo describe(const cublasLtMatmulAlgo_t& algo) {
  AlgoInfo a;
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &a.id, sizeof(a.id), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &a.tile, sizeof(a.tile), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &a.splitk, sizeof(a.splitk), &w);
  cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &a.reduction, sizeof(a.reduction), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &a.swizzle, sizeof(a.swizzle), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &a.custom, sizeof(a.custom), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &a.stages, sizeof(a.stages), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &a.inner, sizeof(a.inner), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &a.cluster, sizeof(a.cluster), &w);
  return a;
}

std::string algo_str(const AlgoInfo& a) {
  char buf[128];
  std::snprintf(
        buf, sizeof(buf), "id %2d tile %3u stages %2u inner %u cluster %2u swz %u opt %u splitk %d red %u", a.id,
        a.tile, a.stages, a.inner, a.cluster, a.swizzle, a.custom, a.splitk, a.reduction);
  return buf;
}

struct Result {
  std::string name;
  Timing t;
  double max_err;
  size_t workspace;
  bool deterministic;
};

void bench_shape(const Shape& s, int max_algos) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  const auto out_dtype = s.f32_out ? torch::kFloat32 : torch::kBFloat16;
  const auto a = torch::randn({s.M, s.K}, opts), b = torch::randn({s.N, s.K}, opts);
  const auto qa = quantize_mx(a, true, false), qb = quantize_mx(b, true, false);
  const int64_t flops = 2 * s.M * s.N * s.K;

  // torch's choice
  auto ref = at::_scaled_mm(qa.data, qb.data.t(), qa.inv_scale, qb.inv_scale, {}, {}, out_dtype, false);
  const auto t_torch = time_calls(
        [&] {
          at::_scaled_mm_out(ref, qa.data, qb.data.t(), qa.inv_scale, qb.inv_scale, {}, {}, out_dtype, false);
        },
        flops);
  const auto ref32 = ref.to(torch::kFloat32);

  // the same GEMM through cublasLt: D^T (N, M) col-major = B (K-major, op T) . A^T (K-major, op N)
  const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  cublasLtMatmulDesc_t desc;
  LT_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));
  const void* a_scale = qb.inv_scale.data_ptr();
  const void* b_scale = qa.inv_scale.data_ptr();
  LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
  const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
  cublasLtMatrixLayout_t la, lb, lc;
  LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, s.K, s.N, s.K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, s.K, s.M, s.K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lc, s.f32_out ? CUDA_R_32F : CUDA_R_16BF, s.N, s.M, s.N));

  auto out = torch::empty_like(ref);
  const size_t ws_bytes = size_t{256} << 20;
  auto workspace = torch::empty({static_cast<int64_t>(ws_bytes)}, opts.dtype(torch::kUInt8));
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

  std::vector<cublasLtMatmulHeuristicResult_t> heur(static_cast<size_t>(max_algos));
  int n_heur = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, lc, pref, max_algos, heur.data(), &n_heur));

  const float alpha = 1.f, beta = 0.f;
  auto run = [&](const cublasLtMatmulAlgo_t& algo, size_t ws) {
    LT_CHECK(cublasLtMatmul(
          handle, desc, &alpha, qb.data.data_ptr(), la, qa.data.data_ptr(), lb, &beta, out.data_ptr(), lc,
          out.data_ptr(), lc, &algo, workspace.data_ptr(), ws, at::cuda::getCurrentCUDAStream()));
  };
  std::vector<Result> results;
  auto measure = [&](const std::string& name, const cublasLtMatmulAlgo_t& algo, size_t ws) {
    out.zero_();
    const auto t = time_calls(
          [&] {
            run(algo, ws);
          },
          flops);
    const double err = (out.to(torch::kFloat32) - ref32).abs().max().item<double>();
    const auto first = out.clone();
    run(algo, ws);
    results.push_back({name, t, err, ws, torch::equal(first, out)});
  };
  for (int i = 0; i < n_heur; ++i) {
    if (heur[i].state != CUBLAS_STATUS_SUCCESS)
      continue;
    char tag[16];
    std::snprintf(tag, sizeof(tag), "h%-2d", i);
    measure(tag + algo_str(describe(heur[i].algo)), heur[i].algo, heur[i].workspaceSize);
  }

  // split-K variants of the heuristics' algorithms, where the algorithm supports it
  std::vector<int> seen_ids;
  for (int i = 0; i < n_heur; ++i) {
    if (heur[i].state != CUBLAS_STATUS_SUCCESS)
      continue;
    const auto info = describe(heur[i].algo);
    if (std::find(seen_ids.begin(), seen_ids.end(), info.id) != seen_ids.end())
      continue;
    seen_ids.push_back(info.id);
    int32_t splitk_support = 0;
    uint32_t red_mask = 0;
    size_t w;
    cublasLtMatmulAlgoCapGetAttribute(
          &heur[i].algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_support, sizeof(splitk_support), &w);
    cublasLtMatmulAlgoCapGetAttribute(
          &heur[i].algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &red_mask, sizeof(red_mask), &w);
    std::printf("  algo id %d: splitk support %d, reduction mask %u\n", info.id, splitk_support, red_mask);
    if (!splitk_support)
      continue;
    for (const int32_t splitk : {2, 3, 4, 6, 8})
      for (const uint32_t red :
           {CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE,
            CUBLASLT_REDUCTION_SCHEME_INPLACE}) {
        if (!(red_mask & red))
          continue;
        auto algo = heur[i].algo;
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red));
        cublasLtMatmulHeuristicResult_t check{};
        if (cublasLtMatmulAlgoCheck(handle, desc, la, lb, lc, lc, &algo, &check) != CUBLAS_STATUS_SUCCESS ||
            check.state != CUBLAS_STATUS_SUCCESS || check.workspaceSize > ws_bytes)
          continue;
        measure("sk " + algo_str(describe(algo)), algo, check.workspaceSize);
      }
  }

  std::sort(results.begin(), results.end(), [](const Result& x, const Result& y) {
    return x.t.ms < y.t.ms;
  });
  std::printf(
        "\n== %s: M %ld N %ld K %ld %s | torch: %.1f us, %.0f TFLOP/s | %d heuristics\n", s.what.c_str(), s.M, s.N, s.K,
        s.f32_out ? "f32" : "bf16", t_torch.ms * 1e3, t_torch.tflops, n_heur);
  const size_t show = std::min<size_t>(results.size(), 12);
  for (size_t i = 0; i < show; ++i)
    std::printf(
          "  %8.1f us %6.0f TF/s  ws %6zu KB  err %.3g %s %s\n", results[i].t.ms * 1e3, results[i].t.tflops,
          results[i].workspace >> 10, results[i].max_err, results[i].deterministic ? "det  " : "NONDET",
          results[i].name.c_str());

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(desc);
}

} // namespace

int main(int argc, char** argv) {
  int64_t T = 16384, C = 768, V = 32768;
  int algos = 32;
  int64_t torch_ws_mb = -1;
  bool f32 = false;
  std::vector<Shape> shapes;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&] {
      return std::string(argv[++i]);
    };
    if (arg == "--tokens")
      T = std::stoll(next());
    else if (arg == "--embd")
      C = std::stoll(next());
    else if (arg == "--algos")
      algos = std::stoi(next());
    else if (arg == "--torch-ws-mb")
      torch_ws_mb = std::stoll(next());
    else if (arg == "--f32")
      f32 = true;
    else if (arg == "--shape") {
      int64_t m, n, k;
      if (std::sscanf(next().c_str(), "%ld,%ld,%ld", &m, &n, &k) != 3)
        return std::fprintf(stderr, "bad --shape\n"), 2;
      shapes.push_back({m, n, k, f32, "custom"});
    }
    else
      return std::fprintf(
                   stderr, "usage: bench_gemm [--tokens T] [--embd C] [--algos N] [--f32] [--shape M,N,K] "
                           "[--torch-ws-mb MB]\n"),
             2;
  }
  torch::manual_seed(0);
  const int64_t chunk = std::min<int64_t>(4096, T);
  if (shapes.empty())
    shapes = {
          {T, C, C, false, "c_proj fwd, c_proj dX"},
          {T, C, 3 * C, false, "qkv dX"},
          {T, C, 4 * C, false, "mlp c_proj fwd, c_fc dX"},
          {T, 3 * C, C, false, "qkv fwd"},
          {T, 4 * C, C, false, "c_fc fwd, mlp c_proj dX"},
          {chunk, V, C, false, "lm_head fwd chunk"},
          {chunk, C, V, false, "lm_head dX chunk"},
          {C, C, T, false, "c_proj dW"},
          {3 * C, C, T, false, "qkv dW"},
          {4 * C, C, T, false, "c_fc dW"},
          {C, 4 * C, T, false, "mlp c_proj dW"},
          {V, C, T, true, "lm_head dW"},
    };
  if (torch_ws_mb >= 0)
    at::cuda::setCUDABlasLtWorkspaceSize(static_cast<size_t>(torch_ws_mb) << 20);
  std::printf("torch cublasLt workspace: %zu KB\n", at::cuda::getCUDABlasLtWorkspaceSize() >> 10);
  for (const auto& s : shapes)
    bench_shape(s, algos);
  return 0;
}
