// Times cuBLASLt's algorithms for the MXFP8 GEMM shapes of a training micro-step against at::_scaled_mm's choice and
// CUTLASS (mx_gemm_kernel.h).
// Usage: bench_gemm [--tokens T] [--embd C] [--algos N] [--f32] [--shape M,N,K] [--torch-ws-mb MB] [--fp4]
//                   [--sustain S] [--wgrad S] [--stage2 S] [--dgrad S]
// Also reports whether each algorithm gives the same bits twice (split-K with in-place reduction does not).
// --algos 0 skips cuBLASLt's algorithms. --fp4: also CUTLASS NVFP4 (nvfp4_gemm_kernel.h) per config, on random e2m1
// operands. --sustain S: also CUTLASS MXFP8 and the best NVFP4 config back to back for S seconds each (the power-capped
// steady state); the summary then uses those. --wgrad S: only the weight-gradient shapes, each S seconds as MXFP8
// (mx_gemm_f32) and as NVFP4 from the same MX operands (nvfp4.h: conversion + GEMM), and the NVFP4 GEMM alone.
// --stage2 S: the kernels that write the weight gradients' operands, MX vs NVFP4 transposes (each S seconds).
// --dgrad S: NVFP4 input gradients per layer: the dX GEMMs and gradient quantization, MX vs NVFP4 (each S seconds).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <torch/torch.h>

#include "nanochat/model/fp8.h"
#include "nanochat/model/mx_gemm.h"
#include "nanochat/model/nvfp4.h"
#include "nanochat/model/nvfp4_gemm_kernel.h"

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

// mean ms per call over `seconds` of back-to-back calls
template <class F>
double sustained_ms(F&& f, double seconds) {
  cudaDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  int64_t n = 0;
  double elapsed = 0;
  while (elapsed < seconds) {
    for (int i = 0; i < 20; ++i)
      f();
    cudaDeviceSynchronize();
    n += 20;
    elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  return elapsed * 1e3 / static_cast<double>(n);
}

// host time per call: 200 calls enqueued without syncing
template <class F>
double host_us(F&& f) {
  cudaDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 200; ++i)
    f();
  const auto t1 = std::chrono::steady_clock::now();
  cudaDeviceSynchronize();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / 200;
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

// NVFP4 configs at shape s, after checking each against fp32 (unit scales); returns {MXFP8 ms, best NVFP4 ms},
// sustained if sustain_s > 0
std::pair<double, double> bench_nvfp4(
      const Shape& s, const torch::Tensor& out, double cutlass_ms, const std::function<void()>& cutlass_call,
      double sustain_s) {
  namespace k = nanochat::kernels;
  const auto u8 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8);
  const auto a = torch::randint(0, 256, {s.M, s.K / 2}, u8), b = torch::randint(0, 256, {s.N, s.K / 2}, u8);
  auto a_scale = torch::full({k::nvfp4_scale_bytes(s.M, s.K)}, 0x38, u8); // ue4m3 1.0
  auto b_scale = torch::full({k::nvfp4_scale_bytes(s.N, s.K)}, 0x38, u8);
  const auto lut = torch::tensor(
        {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f},
        torch::TensorOptions().device(torch::kCUDA));
  const auto unpack = [&](const torch::Tensor& p) {
    const auto q = p.to(torch::kLong);
    const auto lo = lut.take(torch::bitwise_and(q, 15)), hi = lut.take(torch::bitwise_right_shift(q, 4));
    return torch::stack({lo, hi}, -1).reshape({p.size(0), -1});
  };
  const auto ref = torch::matmul(unpack(a), unpack(b).t());
  const double ref_max = ref.abs().max().item<double>();
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bool f32 = out.scalar_type() == torch::kFloat32;
  const auto call = [&](int c) {
    const char* error = k::cutlass_nvfp4_gemm(
          c, a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(), out.data_ptr(), f32, s.M, s.N, s.K,
          stream);
    if (error != nullptr) {
      std::fprintf(stderr, "%s\n", error);
      std::exit(1);
    }
  };
  std::vector<double> errs;
  for (int c = 0; c < k::nvfp4_gemm_configs(); ++c) {
    out.zero_();
    call(c);
    errs.push_back((out.to(torch::kFloat32) - ref).abs().max().item<double>() / ref_max);
  }
  // timing: scales in [0.5, 2], as the data's bits move the power draw
  a_scale.random_(0x30, 0x41);
  b_scale.random_(0x30, 0x41);
  const int64_t flops = 2 * s.M * s.N * s.K;
  double best = 1e30;
  int best_c = 0;
  for (int c = 0; c < k::nvfp4_gemm_configs(); ++c) {
    const auto t = time_calls(
          [&] {
            call(c);
          },
          flops);
    if (t.ms < best)
      best = t.ms, best_c = c;
    std::printf(
          "  nvfp4 %-24s %8.1f us %6.0f TF/s  %.2fx mxfp8  rel err %.2g\n", k::nvfp4_gemm_config_name(c), t.ms * 1e3,
          t.tflops, cutlass_ms / t.ms, errs[c]);
  }
  if (sustain_s <= 0)
    return {cutlass_ms, best};
  const double fp8 = sustained_ms(cutlass_call, sustain_s), fp4 = sustained_ms(
                                                                  [&] {
                                                                    call(best_c);
                                                                  },
                                                                  sustain_s);
  std::printf(
        "  sustained %.0f s each: mxfp8 %.1f us, nvfp4 (%s) %.1f us, %.2fx\n", sustain_s, fp8 * 1e3,
        k::nvfp4_gemm_config_name(best_c), fp4 * 1e3, fp8 / fp4);
  return {fp8, fp4};
}

// returns {CUTLASS MXFP8 ms, best NVFP4 ms (0 without fp4)}
std::pair<double, double> bench_shape(const Shape& s, int max_algos, bool fp4, double sustain_s) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  const auto out_dtype = s.f32_out ? torch::kFloat32 : torch::kBFloat16;
  const auto a = torch::randn({s.M, s.K}, opts), b = torch::randn({s.N, s.K}, opts);
  const auto qa = quantize_mx(a, true, false), qb = quantize_mx(b, true, false);
  const int64_t flops = 2 * s.M * s.N * s.K;

  // torch's choice
  auto ref = at::_scaled_mm(qa.data, qb.data.t(), qa.inv_scale, qb.inv_scale, {}, {}, out_dtype, false);
  const auto torch_call = [&] {
    at::_scaled_mm_out(ref, qa.data, qb.data.t(), qa.inv_scale, qb.inv_scale, {}, {}, out_dtype, false);
  };
  const auto t_torch = time_calls(torch_call, flops);
  const double host_torch = host_us(torch_call);
  const auto ref32 = ref.to(torch::kFloat32);

  // CUTLASS (mx_gemm_kernel.h)
  auto out_cutlass = torch::empty_like(ref);
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cutlass);
  const auto cutlass_call = [&] {
    nanochat::mx_gemm_out(out_cutlass, qa.data, qa.inv_scale, qb.data, qb.inv_scale);
  };
  const auto t_cutlass = time_calls(cutlass_call, flops);
  const double host_cutlass = host_us(cutlass_call);
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cublas);
  const double cutlass_err = (out_cutlass.to(torch::kFloat32) - ref32).abs().max().item<double>();

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
  if (max_algos > 0)
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
        "\n== %s: M %ld N %ld K %ld %s | torch: %.1f us, %.0f TFLOP/s (host %.1f us) | cutlass: %.1f us, %.0f "
        "TFLOP/s (host %.1f us), err %.3g | %d heuristics\n",
        s.what.c_str(), s.M, s.N, s.K, s.f32_out ? "f32" : "bf16", t_torch.ms * 1e3, t_torch.tflops, host_torch,
        t_cutlass.ms * 1e3, t_cutlass.tflops, host_cutlass, cutlass_err, n_heur);
  const size_t show = std::min<size_t>(results.size(), 12);
  for (size_t i = 0; i < show; ++i)
    std::printf(
          "  %8.1f us %6.0f TF/s  ws %6zu KB  err %.3g %s %s\n", results[i].t.ms * 1e3, results[i].t.tflops,
          results[i].workspace >> 10, results[i].max_err, results[i].deterministic ? "det  " : "NONDET",
          results[i].name.c_str());

  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cutlass); // for cutlass_call
  const auto times = fp4 ? bench_nvfp4(s, out_cutlass, t_cutlass.ms, cutlass_call, sustain_s)
                         : std::pair<double, double>{t_cutlass.ms, 0};
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cublas);

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(desc);
  return times;
}

// weight gradient (M, N) fp32 += go_t (M, K) . in_t (N, K)^T, both MX along K (tokens); returns {MXFP8, NVFP4 GEMM
// alone} ms
std::pair<double, double> bench_wgrad(const Shape& s, double seconds) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  const auto go = quantize_mx(torch::randn({s.M, s.K}, opts), true, false);
  const auto in = quantize_mx(torch::randn({s.N, s.K}, opts), true, false);
  auto out = torch::zeros({s.M, s.N}, opts.dtype(torch::kFloat32));
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cutlass);
  const double mx = sustained_ms(
        [&] {
          nanochat::mx_gemm_f32(go.data, go.inv_scale, in.data, in.inv_scale, out, true);
        },
        seconds);
  const nanochat::Nvfp4Backward o;
  nanochat::set_nvfp4_backward(&o);
  const double convert = sustained_ms(
        [&] {
          nanochat::nvfp4_grad_weight(go.data, go.inv_scale, {}, in.data, in.inv_scale, {}, out, true);
        },
        seconds);
  const auto g4 = nanochat::mx_to_nvfp4(go.data, go.inv_scale, true, true, 1);
  const auto i4 = nanochat::mx_to_nvfp4(in.data, in.inv_scale, true, false);
  nanochat::set_nvfp4_backward(nullptr);
  const double gemm = sustained_ms(
        [&] {
          nanochat::nvfp4_gemm_f32(g4, i4, out, true);
        },
        seconds);
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cublas);
  std::printf(
        "%-16s M %5ld N %5ld K %5ld | mxfp8 %8.1f us | nvfp4 from mx %8.1f us | nvfp4 gemm %8.1f us | %.2fx\n",
        s.what.c_str(), s.M, s.N, s.K, mx * 1e3, convert * 1e3, gemm * 1e3, mx / gemm);
  return {mx, gemm};
}

// The weight gradients' operand producers at T tokens, width C: MX transposes vs NVFP4 (Hadamard; the gradients also
// stochastic rounding), as training runs them. Returns {MX, NVFP4} ms summed.
std::pair<double, double> bench_stage2(int64_t T, int64_t C, double seconds) {
  using nanochat::Nvfp4Role;
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  const nanochat::Nvfp4Backward o;
  double mx_sum = 0, fp4_sum = 0;
  const auto row = [&](const char* what, auto&& run) {
    nanochat::set_nvfp4_backward(nullptr);
    const double mx = sustained_ms(
          [&] {
            run(Nvfp4Role::None);
          },
          seconds);
    nanochat::set_nvfp4_backward(&o);
    const double fp4 = sustained_ms(
          [&] {
            run(Nvfp4Role::WgradInput);
          },
          seconds);
    const double fp4_sr = sustained_ms(
          [&] {
            run(Nvfp4Role::WgradGrad);
          },
          seconds);
    nanochat::set_nvfp4_backward(nullptr);
    std::printf("%-34s | mx %7.1f us | nvfp4 %7.1f us | nvfp4 sr %7.1f us\n", what, mx * 1e3, fp4 * 1e3, fp4_sr * 1e3);
    return std::tuple{mx, fp4, fp4_sr};
  };
  // quantize_mx (both layouts): attention c_proj's input (T, C) and gradient (T, C), the MLP c_proj's gradient
  const auto x = torch::randn({T, C}, opts);
  const auto [q_mx, q_in, q_sr] = row("quantize_mx (T, C)", [&](Nvfp4Role role) {
    nanochat::quantize_mx(x, true, true, false, Nvfp4Role::None, role);
  });
  mx_sum += 3 * q_mx, fp4_sum += q_in + 2 * q_sr;
  // relu^2 GEMMs (CUTLASS): c_fc's forward (relu(h)^2's transpose: input) and the MLP c_proj's dgrad (dh's: gradient)
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cutlass);
  const auto xq = quantize_mx(x, true, false), wq = quantize_mx(torch::randn({4 * C, C}, opts), true, false);
  auto h = torch::randn({T, 4 * C}, opts);
  const auto [f_mx, f_in, f_sr] = row("relu^2 fwd GEMM (T, 4C, C)", [&](Nvfp4Role role) {
    auto q = nanochat::empty_mx(T, 4 * C, opts, true, true, Nvfp4Role::None, role);
    const auto [out, out_t] = nanochat::mx_outs(q);
    nanochat::mx_gemm_relu_square(
          h, q.data, q.inv_scale, q.data_t, q.inv_scale_t, xq.data, xq.inv_scale, wq.data, wq.inv_scale, out_t.fp4);
    nanochat::finish_fp4(q);
  });
  const auto [b_mx, b_in, b_sr] = row("relu^2 bwd GEMM (T, 4C, C)", [&](Nvfp4Role role) {
    auto q = nanochat::empty_mx(T, 4 * C, opts, true, true, Nvfp4Role::None, role);
    const auto [out, out_t] = nanochat::mx_outs(q);
    nanochat::mx_gemm_relu_square_bwd(
          xq.data, xq.inv_scale, wq.data, wq.inv_scale, h, q.data, q.inv_scale, q.data_t, q.inv_scale_t, out_t.fp4);
    nanochat::finish_fp4(q);
  });
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cublas);
  mx_sum += f_mx + b_mx, fp4_sum += f_in + b_sr;
  std::printf("per layer (quantize x3, relu^2 GEMMs): mx %.1f us, nvfp4 %.1f us\n", mx_sum * 1e3, fp4_sum * 1e3);
  return {mx_sum, fp4_sum};
}

// NVFP4 dgrad per layer at T tokens, width C, NVFP4 wgrad on throughout: the input-gradient GEMMs (the MLP c_proj's
// with its relu^2 epilogue) and quantizing the Linears' output gradients, MX rows vs NVFP4 (stochastic rounding).
// Returns {MX, NVFP4} ms summed.
std::pair<double, double> bench_dgrad(int64_t T, int64_t C, double seconds) {
  using nanochat::Nvfp4Role;
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  const nanochat::Nvfp4Backward wgrad, both{.dgrad = true};
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cutlass);
  double mx_sum = 0, fp4_sum = 0;
  const auto line = [&](const char* what, auto&& mx_run, auto&& fp4_run, int times) {
    nanochat::set_nvfp4_backward(&wgrad);
    const double mx = sustained_ms(mx_run, seconds);
    nanochat::set_nvfp4_backward(&both);
    const double fp4 = sustained_ms(fp4_run, seconds);
    nanochat::set_nvfp4_backward(nullptr);
    std::printf("%-30s | mx %7.1f us | nvfp4 %7.1f us | %.2fx\n", what, mx * 1e3, fp4 * 1e3, mx / fp4);
    mx_sum += times * mx, fp4_sum += times * fp4;
  };
  // plain dX GEMMs: gradient (T, K) . weight^T (C, K)^T
  for (const auto& [what, K] :
       {std::pair{"c_proj dX (T, C, C)", C}, {"qkv dX (T, C, 3C)", 3 * C}, {"c_fc dX (T, C, 4C)", 4 * C}}) {
    const auto g = torch::randn({T, K}, opts), w = torch::randn({C, K}, opts) * 0.05;
    const auto gm = nanochat::quantize_mx(g, true, false), wm = nanochat::quantize_mx(w, true, false);
    nanochat::set_nvfp4_backward(&both);
    const auto g4 = nanochat::quantize_mx(g, true, false, false, Nvfp4Role::DgradGrad);
    const auto w4 = nanochat::quantize_mx(w, true, false, false, Nvfp4Role::DgradWeight);
    line(
          what,
          [&] {
            nanochat::mx_gemm(gm.data, gm.inv_scale, wm.data, wm.inv_scale, torch::kBFloat16);
          },
          [&] {
            nanochat::nvfp4_gemm(g4.nvfp4(), w4.nvfp4());
          },
          1);
  }
  // the MLP c_proj's dX with relu^2's backward: dh rows (dgrad) and transpose (wgrad, NVFP4 either way)
  {
    const auto go = torch::randn({T, C}, opts), w_t = torch::randn({4 * C, C}, opts) * 0.05;
    const auto h = torch::randn({T, 4 * C}, opts);
    const auto gm = nanochat::quantize_mx(go, true, false), wm = nanochat::quantize_mx(w_t, true, false);
    nanochat::set_nvfp4_backward(&both);
    const auto g4 = nanochat::quantize_mx(go, true, false, false, Nvfp4Role::DgradGrad);
    const auto w4 = nanochat::quantize_mx(w_t, true, false, false, Nvfp4Role::DgradWeight);
    const auto run = [&](bool fp4) {
      auto q = nanochat::empty_mx(
            T, 4 * C, opts, true, true, fp4 ? Nvfp4Role::DgradGrad : Nvfp4Role::None, Nvfp4Role::WgradGrad);
      const auto [out, out_t] = nanochat::mx_outs(q);
      if (fp4)
        nanochat::nvfp4_gemm_relu_square_bwd(g4.nvfp4(), w4.nvfp4(), h, out, out_t);
      else
        nanochat::mx_gemm_relu_square_bwd(
              gm.data, gm.inv_scale, wm.data, wm.inv_scale, h, q.data, q.inv_scale, q.data_t, q.inv_scale_t, out_t.fp4);
      nanochat::finish_fp4(q);
    };
    line(
          "relu^2 dX (T, 4C, C)",
          [&] {
            run(false);
          },
          [&] {
            run(true);
          },
          1);
  }
  // the output gradients (T, C) of attention's and the MLP's c_proj (qkv's comes from the attention kernels)
  const auto x = torch::randn({T, C}, opts);
  line(
        "quantize grad (T, C)",
        [&] {
          nanochat::quantize_mx(x, true, true, false, Nvfp4Role::None, Nvfp4Role::WgradGrad);
        },
        [&] {
          nanochat::quantize_mx(x, true, true, false, Nvfp4Role::DgradGrad, Nvfp4Role::WgradGrad);
        },
        2);
  nanochat::set_mx_gemm_backend(nanochat::MxGemmBackend::Cublas);
  std::printf(
        "per layer (dX GEMMs, quantize x2): mx %.1f us, nvfp4 %.1f us (%.2fx)\n", mx_sum * 1e3, fp4_sum * 1e3,
        mx_sum / fp4_sum);
  return {mx_sum, fp4_sum};
}

} // namespace

int main(int argc, char** argv) {
  int64_t T = 16384, C = 768, V = 32768;
  int algos = 32;
  int64_t torch_ws_mb = -1;
  bool f32 = false, fp4 = false;
  double sustain_s = 0, wgrad_s = 0, stage2_s = 0, dgrad_s = 0;
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
    else if (arg == "--fp4")
      fp4 = true;
    else if (arg == "--sustain")
      sustain_s = std::stod(next());
    else if (arg == "--wgrad")
      wgrad_s = std::stod(next());
    else if (arg == "--stage2")
      stage2_s = std::stod(next());
    else if (arg == "--dgrad")
      dgrad_s = std::stod(next());
    else if (arg == "--shape") {
      int64_t m, n, k;
      if (std::sscanf(next().c_str(), "%ld,%ld,%ld", &m, &n, &k) != 3)
        return std::fprintf(stderr, "bad --shape\n"), 2;
      shapes.push_back({m, n, k, f32, "custom"});
    }
    else
      return std::fprintf(
                   stderr, "usage: bench_gemm [--tokens T] [--embd C] [--algos N] [--f32] [--shape M,N,K] "
                           "[--torch-ws-mb MB] [--fp4] [--sustain S] [--wgrad S] [--stage2 S] [--dgrad S]\n"),
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
  if (stage2_s > 0) {
    bench_stage2(T, C, stage2_s);
    return 0;
  }
  if (dgrad_s > 0) {
    bench_dgrad(T, C, dgrad_s);
    return 0;
  }
  if (wgrad_s > 0) {
    double mx = 0, fp4 = 0;
    for (const auto& s : shapes)
      if (s.what.ends_with("dW") && s.what != "lm_head dW") { // lm_head's stays MXFP8
        const auto [a, b] = bench_wgrad(s, wgrad_s);
        mx += a, fp4 += b;
      }
    std::printf("block dW shapes: mxfp8 %.1f us, nvfp4 gemm %.1f us (%.2fx)\n", mx * 1e3, fp4 * 1e3, mx / fp4);
    return 0;
  }
  if (torch_ws_mb >= 0)
    at::cuda::setCUDABlasLtWorkspaceSize(static_cast<size_t>(torch_ws_mb) << 20);
  std::printf("torch cublasLt workspace: %zu KB\n", at::cuda::getCUDABlasLtWorkspaceSize() >> 10);
  double fp8_ms = 0, fp4_ms = 0;
  for (const auto& s : shapes) {
    const auto [a, b] = bench_shape(s, algos, fp4, sustain_s);
    fp8_ms += a;
    fp4_ms += b;
  }
  if (fp4)
    std::printf(
          "\nall shapes: cutlass mxfp8 %.1f us, best nvfp4 %.1f us (%.2fx)\n", fp8_ms * 1e3, fp4_ms * 1e3,
          fp8_ms / fp4_ms);
  return 0;
}
