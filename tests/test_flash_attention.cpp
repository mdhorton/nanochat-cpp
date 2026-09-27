// FlashAttention-2 built from external/flash-attention vs PyTorch's built-in copy and an fp32 reference.
#include <gtest/gtest.h>

#include <ATen/cuda/CUDAEvent.h>

#include "nanochat/model/flash_attention.h"

using namespace nanochat;

namespace {

struct Result {
  torch::Tensor out, dq, dk, dv;
};

enum class Impl { Ours, PyTorch, Reference };

// PyTorch: at::_flash_attention_forward, as gpt.cpp's fa2_attention; Reference: fp32 SDPA with a mask
torch::Tensor run_forward(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window, Impl impl) {
  const int64_t T = q.size(1);
  const bool local = window >= 0 && window < T;
  torch::Tensor out;
  if (impl == Impl::Ours)
    out = flash_attention(q, k, v, window);
  else if (impl == Impl::PyTorch)
    out = std::get<0>(at::_flash_attention_forward(
          q, k, v, std::nullopt, std::nullopt, T, T, 0.0, true, false, std::nullopt,
          local ? std::optional<int64_t>(window) : std::nullopt, local ? std::optional<int64_t>(0) : std::nullopt));
  else {
    const auto opts = torch::TensorOptions().device(q.device()).dtype(torch::kInt64);
    const auto row = torch::arange(T, opts).unsqueeze(1), col = torch::arange(T, opts).unsqueeze(0);
    auto mask = col <= row;
    if (local)
      mask = mask & ((row - col) <= window);
    out = at::scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask)
                .transpose(1, 2);
  }
  return out;
}

// forward and backward of out * g
Result run(
      const torch::Tensor& q_in, const torch::Tensor& k_in, const torch::Tensor& v_in, const torch::Tensor& g,
      int64_t window, Impl impl) {
  const auto leaf = [&](const torch::Tensor& t) {
    return (impl == Impl::Reference ? t.to(torch::kFloat32) : t).detach().requires_grad_();
  };
  auto q = leaf(q_in), k = leaf(k_in), v = leaf(v_in);
  const auto out = run_forward(q, k, v, window, impl);
  out.backward(g.to(out.scalar_type()));
  return {out.detach(), q.grad(), k.grad(), v.grad()};
}

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-30)).item<double>();
}

} // namespace

// The same algorithm as PyTorch's copy, but not bit-identical (FlashAttention version, build flags): at least as
// close to fp32 as PyTorch's, and deterministic but for dq (summed with atomics).
TEST(FlashAttention, MatchesPyTorch) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  for (const auto& [T, window] : {std::pair<int64_t, int64_t>{512, -1}, {512, 128}, {2048, 512}, {300, 64}}) {
    const int64_t B = 2, H = 3, D = 128;
    // q, k contiguous; v a strided view (as a narrowed qkv)
    const auto q = torch::randn({B, T, H, D}, opts), k = torch::randn({B, T, H, D}, opts);
    const auto v = torch::randn({B, T, 2 * H, D}, opts).narrow(2, H, H);
    const auto g = torch::randn({B, T, H, D}, opts);
    const auto a = run(q, k, v, g, window, Impl::Ours), b = run(q, k, v, g, window, Impl::PyTorch);
    const auto r = run(q, k, v, g, window, Impl::Reference);
    const auto name = "T=" + std::to_string(T) + " window=" + std::to_string(window);
    for (const auto& [x, y, z, what] :
         {std::tuple{a.out, b.out, r.out, "out"},
          {a.dq, b.dq, r.dq, "dq"},
          {a.dk, b.dk, r.dk, "dk"},
          {a.dv, b.dv, r.dv, "dv"}}) {
      EXPECT_LT(rel_diff(x, z), 1e-2) << name << " " << what;
      EXPECT_LE(rel_diff(x, z), rel_diff(y, z) * 1.1) << name << " " << what;
    }
    const auto a2 = run(q, k, v, g, window, Impl::Ours);
    EXPECT_TRUE(torch::equal(a.out, a2.out) && torch::equal(a.dk, a2.dk) && torch::equal(a.dv, a2.dv)) << name;
  }
}

// Timing at d12's shapes (not a check): --gtest_also_run_disabled_tests --gtest_filter=FlashAttention.DISABLED_Speed
TEST(FlashAttention, DISABLED_Speed) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  const int64_t B = 8, T = 2048, H = 6, D = 128;
  const auto q = torch::randn({B, T, H, D}, opts), k = torch::randn({B, T, H, D}, opts);
  const auto v = torch::randn({B, T, H, D}, opts), g = torch::randn({B, T, H, D}, opts);
  for (const int64_t window : {-1, 512})
    for (const auto impl : {Impl::PyTorch, Impl::Ours}) {
      const auto time_ms = [&](const auto& fn) {
        for (int i = 0; i < 3; ++i)
          fn();
        auto start = at::cuda::CUDAEvent(cudaEventDefault), end = at::cuda::CUDAEvent(cudaEventDefault);
        start.record();
        for (int i = 0; i < 20; ++i)
          fn();
        end.record();
        end.synchronize();
        return start.elapsed_time(end) / 20;
      };
      const auto fwd = time_ms([&] {
        torch::NoGradGuard no_grad;
        run_forward(q, k, v, window, impl);
      });
      const auto both = time_ms([&] {
        run(q, k, v, g, window, impl);
      });
      std::cout << (impl == Impl::Ours ? "ours    " : "pytorch ") << "window " << window << ": fwd " << fwd * 1e3
                << " us, fwd+bwd " << both * 1e3 << " us\n";
    }
}
