// Times the attention backends at a training micro-step's shapes: the FA2 path the model uses (fa2_attention in
// gpt.cpp) against SDPA's cuDNN and memory-efficient kernels, forward and forward + backward, full causal and
// sliding window (SDPA takes the window as an explicit mask; FA2 natively). Checks each against FA2. Full causal also
// times SageAttention2's FP8 forward kernel (a probe for an FP8 attention on sm120). mx: our MXFP8 forward
// (mx_flash.h) with FA2's backward; mx-kernel: its attention kernel without the quantization.
// Usage: bench_attention [--batch B] [--seq T] [--heads H] [--kv-heads Hkv] [--head-dim D] [--window W]
//                        [--depth L] [--pattern SSSL] [--iters N]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "nanochat/model/mx_attention.h"
#include "nanochat/model/mx_flash.h"
#include "nanochat/model/mx_flash_kernel.h"

// SageAttention2 FP8 forward (external/SageAttention/csrc/qattn): q, k int8, v fp8 V^T, P·V accumulated in f32 or f16
#define SAGE_ARGS                                                                                                      \
  torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output, torch::Tensor query_scale,        \
        torch::Tensor key_scale, torch::Tensor value_scale, int tensor_layout, int is_causal, int qk_quant_gran,       \
        float sm_scale, int return_lse
torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(SAGE_ARGS);
torch::Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(SAGE_ARGS);
#undef SAGE_ARGS

namespace {

struct Args {
  int64_t batch = 8, seq = 2048, heads = 6, kv_heads = 6, head_dim = 128, window = -1, depth = 12;
  std::string pattern = "SSSL";
  int iters = 20;
};

// median ms of `iters` timed calls after warmup
template <class F>
double time_ms(F&& f, int iters, int warmup = 5) {
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
  return ms[iters / 2];
}

// q, k, v, out: (B, T, H, D) as the model holds them. window: keys to the left of the query, < 0 for full causal.
using Attention = std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>;

torch::Tensor fa2(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window) {
  const int64_t T = q.size(1);
  std::optional<int64_t> left, right;
  if (window >= 0 && window < T) {
    left = window;
    right = 0;
  }
  return std::get<0>(at::_flash_attention_forward(
        q, k, v, std::nullopt, std::nullopt, T, T, 0.0, /*is_causal=*/true, false, std::nullopt, left, right));
}

// SDPA with only `backend` enabled; the window as a boolean mask (sdpa_attention in gpt.cpp)
struct SdpBackend {
  const char* name;
  void (at::Context::*enable)(bool);
};

const SdpBackend kBackends[] = {
      {"sdpa-flash", &at::Context::setSDPUseFlash},
      {"cudnn", &at::Context::setSDPUseCuDNN},
      {"efficient", &at::Context::setSDPUseMemEfficient},
};

void only(const SdpBackend& b) {
  auto& ctx = at::globalContext();
  ctx.setSDPUseFlash(false);
  ctx.setSDPUseCuDNN(false);
  ctx.setSDPUseMemEfficient(false);
  ctx.setSDPUseMath(false);
  (ctx.*b.enable)(true);
}

torch::Tensor sdpa(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window) {
  const int64_t T = q.size(1);
  const bool gqa = q.size(2) != k.size(2);
  auto qt = q.transpose(1, 2), kt = k.transpose(1, 2), vt = v.transpose(1, 2);
  torch::Tensor out;
  if (window < 0 || window >= T)
    out = at::scaled_dot_product_attention(qt, kt, vt, {}, 0.0, /*is_causal=*/true, std::nullopt, gqa);
  else {
    const auto opts = torch::TensorOptions().device(q.device()).dtype(torch::kInt64);
    const auto row = torch::arange(T, opts).unsqueeze(1), col = torch::arange(T, opts).unsqueeze(0);
    const auto mask = (col <= row) & ((row - col) <= window);
    out = at::scaled_dot_product_attention(qt, kt, vt, mask, 0.0, false, std::nullopt, gqa);
  }
  return out.transpose(1, 2).contiguous();
}

// causal (window < 0) or windowed attention flops for one forward: 4 * B * H * D per (query, key) pair
double fwd_flops(const Args& a, int64_t window) {
  const double T = static_cast<double>(a.seq);
  const double w = window < 0 || window >= a.seq ? T : static_cast<double>(window + 1);
  const double pairs = w >= T ? T * (T + 1) / 2 : w * T - w * (w - 1) / 2;
  return 4.0 * a.batch * a.heads * a.head_dim * pairs;
}

struct Result {
  std::string name;
  double fwd_ms = 0, fwd_bwd_ms = 0;
  double err_out = 0, err_dq = 0, err_dk = 0, err_dv = 0;
  bool ok = true;
  std::string why;
};

double max_diff(const torch::Tensor& a, const torch::Tensor& b) {
  return (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs().max().item<double>();
}

Result bench(
      const std::string& name, const Attention& attend, const torch::Tensor& q, const torch::Tensor& k,
      const torch::Tensor& v, const torch::Tensor& g, std::vector<torch::Tensor>& ref, int iters) {
  Result r{name};
  try {
    torch::Tensor out;
    {
      torch::NoGradGuard no_grad;
      out = attend(q, k, v);
      r.fwd_ms = time_ms(
            [&] {
              out = attend(q, k, v);
            },
            iters);
    }
    auto qg = q.detach().requires_grad_(), kg = k.detach().requires_grad_(), vg = v.detach().requires_grad_();
    std::vector<torch::Tensor> grads;
    const auto step = [&] {
      const auto o = attend(qg, kg, vg);
      grads = torch::autograd::grad({o}, {qg, kg, vg}, {g});
    };
    r.fwd_bwd_ms = time_ms(step, iters);
    if (ref.empty()) // the first backend is the reference
      ref = {out, grads[0], grads[1], grads[2]};
    r.err_out = max_diff(out, ref[0]);
    r.err_dq = max_diff(grads[0], ref[1]);
    r.err_dk = max_diff(grads[1], ref[2]);
    r.err_dv = max_diff(grads[2], ref[3]);
  }
  catch (const std::exception& e) {
    r.ok = false;
    r.why = e.what();
    if (const auto nl = r.why.find('\n'); nl != std::string::npos)
      r.why.resize(nl);
  }
  return r;
}

// SageAttention2's quantization (quant.py) in torch ops, HND: q int8 per 32 rows, k int8 per 64 rows, v fp8 per
// channel over V^T (B, H, D, T), its tokens permuted per 16 so P's accumulator is the next mma's A operand.
struct SageInputs {
  torch::Tensor q, k, v, q_scale, k_scale, v_scale;
};

SageInputs sage_quantize(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double v_max) {
  const auto int8_rows = [](const torch::Tensor& x, int64_t rows) { // x (B, T, H, D)
    const auto t = x.to(torch::kFloat32).transpose(1, 2).contiguous();
    const int64_t B = t.size(0), H = t.size(1), T = t.size(2), D = t.size(3);
    const auto blocks = t.view({B, H, T / rows, rows * D});
    const auto scale = blocks.abs().amax(-1).clamp_min(1e-8) / 127;
    const auto xi = (blocks / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch::kInt8).view({B, H, T, D});
    return std::pair{xi, scale.contiguous()};
  };
  TORCH_CHECK(q.size(1) % 128 == 0, "sage probe needs T % 128 == 0");
  SageInputs s;
  std::tie(s.q, s.q_scale) = int8_rows(q, 32);
  std::tie(s.k, s.k_scale) = int8_rows(k, 64);
  const int64_t T = v.size(1);
  static const int64_t perm[16] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
  std::vector<int64_t> idx(T);
  for (int64_t t = 0; t < T; ++t)
    idx[t] = t / 16 * 16 + perm[t % 16];
  const auto vt = v.to(torch::kFloat32)
                        .permute({0, 2, 3, 1})
                        .index_select(
                              3, torch::tensor(idx, torch::TensorOptions().dtype(torch::kInt64)).to(v.device()));
  s.v_scale = (vt.abs().amax(-1).clamp_min(1e-8) / v_max).contiguous();
  s.v = (vt / s.v_scale.unsqueeze(-1)).to(torch::kFloat8_e4m3fn).contiguous();
  return s;
}

// MXFP8 attention kernel alone, inputs quantized beforehand (as bench_sage)
Result bench_mx_kernel(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
      const torch::Tensor& ref_out, int iters) {
  Result r{"mx-kernel"};
  torch::NoGradGuard no_grad;
  const auto [qd, qs] = nanochat::mx_flash_quantize_rows(q);
  const auto [kd, ks] = nanochat::mx_flash_quantize_rows(k);
  const auto [vd, vs] = nanochat::mx_flash_quantize_vt(v);
  auto out = torch::empty_like(q);
  auto lse = torch::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(torch::kFloat32));
  const auto run = [&] {
    nanochat::kernels::mx_flash_fwd(
          qd.data_ptr(), reinterpret_cast<const uint32_t*>(qs.data_ptr<int32_t>()), kd.data_ptr(),
          reinterpret_cast<const uint32_t*>(ks.data_ptr<int32_t>()), vd.data_ptr(), vs.data_ptr<uint8_t>(),
          out.data_ptr(), lse.data_ptr<float>(), static_cast<int>(q.size(0)), q.size(1), static_cast<int>(q.size(2)),
          static_cast<int>(k.size(2)), window, at::cuda::getCurrentCUDAStream().stream());
  };
  run();
  r.fwd_ms = time_ms(run, iters);
  r.err_out = max_diff(out, ref_out);
  r.why = "fwd only, no quantize";
  return r;
}

// forward only, full causal; the kernel alone is timed (as SageAttention reports), not the quantization
Result bench_sage(
      const std::string& name, bool f16_accum, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
      const torch::Tensor& ref_out, int iters) {
  Result r{name};
  try {
    torch::NoGradGuard no_grad;
    const auto s = sage_quantize(q, k, v, f16_accum ? 2.25 : 448.0);
    auto out = torch::empty({q.size(0), q.size(2), q.size(1), q.size(3)}, q.options());
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(q.size(3)));
    const auto run = [&] {
      (f16_accum ? qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
                 : qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf)(
            s.q, s.k, s.v, out, s.q_scale, s.k_scale, s.v_scale, /*HND*/ 1, /*causal*/ 1, /*per warp*/ 2, sm_scale, 0);
    };
    run();
    r.fwd_ms = time_ms(run, iters);
    r.err_out = max_diff(out.transpose(1, 2), ref_out);
    r.why = "fwd only";
  }
  catch (const std::exception& e) {
    r.ok = false;
    r.why = e.what();
    if (const auto nl = r.why.find('\n'); nl != std::string::npos)
      r.why.resize(nl);
  }
  return r;
}

// The attention's MX inputs: standalone quantization of q, k, v vs the extra time mx_attention_inputs takes to
// write them (rotary norm and value mix), with and without a value embedding. n_embd = heads * head_dim.
void bench_quantization(const Args& a) {
  torch::NoGradGuard no_grad;
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const int64_t B = a.batch, T = a.seq, D = a.head_dim, N = B * T, C = a.heads * D, gate_cols = 12;
  const auto freqs = torch::outer(torch::arange(T, opts), torch::rand({D / 2}, opts));
  const auto cos = freqs.cos().to(torch::kBFloat16).view({1, T, 1, D / 2});
  const auto sin = freqs.sin().to(torch::kBFloat16).view({1, T, 1, D / 2});
  const auto x = torch::randn({N, C}, opts).to(torch::kBFloat16);
  const auto wq = torch::randn({a.heads * D, C}, opts) * 0.03, wk = torch::randn({a.kv_heads * D, C}, opts) * 0.03;
  const auto wv = torch::randn({a.kv_heads * D, C}, opts) * 0.03, wg = torch::randn({a.kv_heads, gate_cols}, opts);
  const auto ve = torch::randn({N, a.kv_heads * D}, opts).to(torch::kBFloat16);
  std::printf("\nattention inputs' MX quantization (ms per layer):\n");
  for (const bool with_ve : {false, true}) {
    const auto inputs = [&](bool quantize) {
      return nanochat::mx_attention_inputs(
            x, wq, wk, wv, cos, sin, 1.2, with_ve ? ve : torch::Tensor(), with_ve ? wg : torch::Tensor(), D, nullptr,
            nullptr, quantize);
    };
    const auto out = inputs(false);
    const auto q = out[0].view({B, T, a.heads, D}), k = out[1].view({B, T, a.kv_heads, D});
    const auto v = out[2].view({B, T, a.kv_heads, D});
    const double standalone = time_ms(
          [&] {
            nanochat::mx_flash_quantize_rows(q);
            nanochat::mx_flash_quantize_rows(k);
            nanochat::mx_flash_quantize_vt(v);
          },
          a.iters);
    const double plain = time_ms(
                       [&] {
                         inputs(false);
                       },
                       a.iters),
                 fused = time_ms(
                       [&] {
                         inputs(true);
                       },
                       a.iters);
    std::printf(
          "  %-8s standalone %.3f, fused %.3f (mx_attention_inputs %.3f -> %.3f)\n", with_ve ? "ve" : "no ve",
          standalone, fused - plain, plain, fused);
  }
}

} // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const auto flag = [&](const char* f) {
      return std::strcmp(argv[i], f) == 0 && i + 1 < argc;
    };
    if (flag("--batch"))
      a.batch = std::atoll(argv[++i]);
    else if (flag("--seq"))
      a.seq = std::atoll(argv[++i]);
    else if (flag("--heads"))
      a.heads = std::atoll(argv[++i]);
    else if (flag("--kv-heads"))
      a.kv_heads = std::atoll(argv[++i]);
    else if (flag("--head-dim"))
      a.head_dim = std::atoll(argv[++i]);
    else if (flag("--window"))
      a.window = std::atoll(argv[++i]);
    else if (flag("--depth"))
      a.depth = std::atoll(argv[++i]);
    else if (flag("--pattern"))
      a.pattern = argv[++i];
    else if (flag("--iters"))
      a.iters = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "unknown flag %s\n", argv[i]);
      return 1;
    }
  }
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  const auto q = torch::randn({a.batch, a.seq, a.heads, a.head_dim}, opts);
  const auto k = torch::randn({a.batch, a.seq, a.kv_heads, a.head_dim}, opts);
  const auto v = torch::randn({a.batch, a.seq, a.kv_heads, a.head_dim}, opts);
  const auto g = torch::randn({a.batch, a.seq, a.heads, a.head_dim}, opts);

  // windows as window_sizes in gpt.cpp: L = seq, S = seq / 4 rounded up to 128
  const int64_t short_window = (a.seq / 4 + 127) / 128 * 128;
  std::vector<int64_t> windows = a.window >= 0 ? std::vector<int64_t>{a.window}
                                               : std::vector<int64_t>{short_window, -1};
  int64_t n_short = 0;
  for (int64_t i = 0; i < a.depth; ++i)
    if (i + 1 < a.depth && std::toupper(a.pattern[i % a.pattern.size()]) == 'S')
      ++n_short;
  std::printf(
        "B %ld T %ld H %ld Hkv %ld D %ld, %d iters (median); layers: %ld short (window %ld), %ld long\n", a.batch,
        a.seq, a.heads, a.kv_heads, a.head_dim, a.iters, n_short, short_window, a.depth - n_short);

  std::vector<std::vector<Result>> all;
  for (const int64_t window : windows) {
    std::printf(
          "\nwindow %s: fwd %.1f GFLOP, fwd + bwd %.1f GFLOP (bwd counted as 2.5x fwd)\n",
          window < 0 ? "full causal" : std::to_string(window).c_str(), fwd_flops(a, window) / 1e9,
          3.5 * fwd_flops(a, window) / 1e9);
    std::printf(
          "%-12s %9s %8s %9s %8s   %-9s %-9s %-9s %-9s\n", "backend", "fwd ms", "TFLOPs", "f+b ms", "TFLOPs", "|d out|",
          "|d dq|", "|d dk|", "|d dv|");
    std::vector<Result> results;
    std::vector<torch::Tensor> ref;
    results.push_back(bench(
          "fa2",
          [&](const auto& q, const auto& k, const auto& v) {
            return fa2(q, k, v, window);
          },
          q, k, v, g, ref, a.iters));
    for (const auto& b : kBackends) {
      only(b);
      auto r = bench(
            b.name,
            [&](const auto& q, const auto& k, const auto& v) {
              return sdpa(q, k, v, window);
            },
            q, k, v, g, ref, a.iters);
      if (r.ok) {
        // what SDPA dispatched to (SDPBackend: 0 math, 1 flash, 2 efficient, 3 cudnn); a mask rules out flash
        const auto qt = q.transpose(1, 2), kt = k.transpose(1, 2), vt = v.transpose(1, 2);
        const bool windowed = window >= 0 && window < a.seq;
        const auto mask = windowed ? std::optional(torch::ones({a.seq, a.seq}, opts.dtype(torch::kBool)))
                                   : std::nullopt;
        const int64_t choice = at::_fused_sdp_choice(
              qt, kt, vt, mask, 0.0, !windowed, std::nullopt, a.heads != a.kv_heads);
        static const char* names[] = {"math", "flash", "efficient", "cudnn"};
        if (choice < 0 || choice > 3 || std::strncmp(b.name, names[choice], 4) != 0)
          r.why = std::string("ran ") + (choice >= 0 && choice <= 3 ? names[choice] : "?");
      }
      results.push_back(std::move(r));
    }
    results.push_back(bench(
          "mx",
          [&](const auto& q, const auto& k, const auto& v) {
            return nanochat::mx_flash_attention(q, k, v, window);
          },
          q, k, v, g, ref, a.iters));
    results.push_back(bench_mx_kernel(q, k, v, window, ref[0], a.iters));
    if (window < 0 || window >= a.seq) {
      results.push_back(bench_sage("sage-f32acc", false, q, k, v, ref[0], a.iters));
      results.push_back(bench_sage("sage-f16acc", true, q, k, v, ref[0], a.iters));
    }
    for (const auto& r : results) {
      if (!r.ok) {
        std::printf("%-12s failed: %s\n", r.name.c_str(), r.why.c_str());
        continue;
      }
      const double f = fwd_flops(a, window);
      if (r.fwd_bwd_ms == 0) {
        std::printf(
              "%-12s %9.3f %8.1f %9s %8s   %-9.3g %s\n", r.name.c_str(), r.fwd_ms, f / r.fwd_ms / 1e9, "-", "-",
              r.err_out, r.why.c_str());
        continue;
      }
      std::printf(
            "%-12s %9.3f %8.1f %9.3f %8.1f   %-9.3g %-9.3g %-9.3g %-9.3g %s\n", r.name.c_str(), r.fwd_ms,
            f / r.fwd_ms / 1e9, r.fwd_bwd_ms, 3.5 * f / r.fwd_bwd_ms / 1e9, r.err_out, r.err_dq, r.err_dk, r.err_dv,
            r.why.c_str());
    }
    all.push_back(std::move(results));
  }

  bench_quantization(a);

  if (windows.size() == 2) {
    std::printf("\nper micro-step (fwd + bwd, %ld short + %ld long layers):\n", n_short, a.depth - n_short);
    double best_mix = 0;
    std::string mix;
    for (const auto& s : all[0]) {
      if (!s.ok || s.fwd_bwd_ms == 0)
        continue;
      for (const auto& l : all[1]) {
        if (!l.ok || l.fwd_bwd_ms == 0)
          continue;
        const double ms = n_short * s.fwd_bwd_ms + (a.depth - n_short) * l.fwd_bwd_ms;
        if (s.name == l.name)
          std::printf("  %-12s %8.2f ms\n", s.name.c_str(), ms);
        if (mix.empty() || ms < best_mix) {
          best_mix = ms;
          mix = s.name + " short / " + l.name + " long";
        }
      }
    }
    std::printf("  best mix: %s, %.2f ms\n", mix.c_str(), best_mix);
  }
  return 0;
}
