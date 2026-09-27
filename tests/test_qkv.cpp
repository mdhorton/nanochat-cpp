// Merged q/k/v (fp8_qkv) vs three fp8_matmuls, and the rotary norm on strided q/k views.
#include <gtest/gtest.h>

#include "nanochat/model/fp8.h"
#include "nanochat/model/gpt.h"
#include "nanochat/model/rotary_norm.h"

using namespace nanochat;

namespace {

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-30)).item<double>();
}

double rel_norm_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).norm() / (b64.norm() + 1e-30)).item<double>();
}

double mismatch(const torch::Tensor& a, const torch::Tensor& b) {
  return a.ne(b).to(torch::kFloat64).mean().item<double>();
}

} // namespace

TEST(Qkv, Fp8MatchesSeparateMatmuls) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const int64_t N = 624, C = 256, gate_cols = 12;
  for (const auto& [nq, nkv] : {std::pair<int64_t, int64_t>{256, 256}, {256, 128}}) { // MHA, GQA
    const auto x_in = torch::randn({N, C}, opts).to(torch::kBFloat16);
    std::vector<torch::Tensor> w_in, up;
    for (const auto n : {nq, nkv, nkv}) {
      w_in.push_back(torch::randn({n, C}, opts) * 0.05 * (1 + static_cast<double>(w_in.size()))); // distinct scales
      up.push_back(torch::randn({N, n}, opts).to(torch::kBFloat16) * (1 + static_cast<double>(up.size())));
    }
    const auto up_gate = torch::randn({N, gate_cols}, opts).to(torch::kBFloat16);

    // fused
    auto x = x_in.clone().requires_grad_();
    std::vector<torch::Tensor> w;
    for (const auto& t : w_in)
      w.push_back(t.clone().requires_grad_());
    const auto out = fp8_qkv(x, w[0], w[1], w[2], gate_cols);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_TRUE(torch::equal(out[3], x_in.narrow(1, 0, gate_cols)));
    auto loss = (out[3].to(torch::kFloat32) * up_gate).sum();
    for (int i = 0; i < 3; ++i)
      loss = loss + (out[i].to(torch::kFloat32) * up[i]).sum();
    loss.backward();

    // three fp8_matmuls, each with its own input leaf (per-branch dx)
    torch::Tensor dx_ref = torch::zeros({N, C}, opts.dtype(torch::kFloat64));
    dx_ref.narrow(1, 0, gate_cols).add_(up_gate.to(torch::kFloat64));
    for (int i = 0; i < 3; ++i) {
      auto xi = x_in.clone().requires_grad_();
      auto wi = w_in[i].clone().requires_grad_();
      const auto o = fp8_matmul(xi, wi);
      (o.to(torch::kFloat32) * up[i]).sum().backward();
      const auto n = std::to_string(nkv) + " part " + std::to_string(i);
      EXPECT_TRUE(torch::equal(out[i], o)) << n;
      // the merged GEMM accumulates in another order: rare 1-ulp differences
      EXPECT_LT(mismatch(w[i].grad(), wi.grad()), 1e-4) << n;
      EXPECT_LT(rel_diff(w[i].grad(), wi.grad()), 4e-3) << n;
      dx_ref += xi.grad().to(torch::kFloat64);
    }
    // the fused dx sums the same three bf16 products in fp32, rounded once
    const auto dx_ref_bf16 = dx_ref.to(torch::kBFloat16);
    EXPECT_LT(mismatch(x.grad(), dx_ref_bf16), 1e-4) << nkv;
    EXPECT_LT(rel_diff(x.grad(), dx_ref), 4e-3) << nkv;
  }
}

TEST(Qkv, RotaryNormOnStridedViews) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const int64_t B = 2, T = 33, H = 3, D = 128;
  const auto freqs = torch::outer(torch::arange(T, opts), torch::rand({D / 2}, opts));
  const auto cos = freqs.cos().to(torch::kBFloat16).view({1, T, 1, D / 2});
  const auto sin = freqs.sin().to(torch::kBFloat16).view({1, T, 1, D / 2});
  const auto qkv = torch::randn({B * T, 3 * H * D}, opts).to(torch::kBFloat16);
  const auto up = torch::randn({B, T, H, D}, opts).to(torch::kBFloat16);
  for (int part = 0; part < 3; ++part) {
    auto base = qkv.clone().requires_grad_();
    const auto view = base.narrow(1, part * H * D, H * D).view({B, T, H, D});
    ASSERT_FALSE(view.is_contiguous());
    auto contig = view.detach().contiguous().requires_grad_();
    const auto a = rotary_rms_norm(view, cos, sin, 1.2), b = rotary_rms_norm(contig, cos, sin, 1.2);
    (a.to(torch::kFloat32) * up).sum().backward();
    (b.to(torch::kFloat32) * up).sum().backward();
    EXPECT_TRUE(torch::equal(a, b)) << part;
    EXPECT_TRUE(torch::equal(base.grad().narrow(1, part * H * D, H * D).view({B, T, H, D}), contig.grad())) << part;
  }
}

// Attention with a value embedding: fused (merged q/k/v, rotary norm) vs the op path. The fused rotary norm rounds
// differently; with FP8 that flips some e5m2 gradient roundings (a 25% step each), so gradients differ by a few %.
TEST(Qkv, AttentionMatchesOpByOp) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const GPTConfig config{.sequence_len = 256, .n_layer = 2, .n_head = 2, .n_kv_head = 2, .n_embd = 256};
  CausalSelfAttention attn(config, 1, opts.dtype(torch::kFloat32));
  ASSERT_TRUE(attn->ve_gate);
  {
    torch::NoGradGuard no_grad;
    for (auto& p : attn->parameters())
      p.normal_(0, 0.05);
  }
  attn->attention = Attention::SDPA; // deterministic
  const int64_t B = 2, T = 208;
  const auto freqs = torch::outer(torch::arange(T, opts), torch::rand({64}, opts));
  const auto cos = freqs.cos().to(torch::kBFloat16).view({1, T, 1, 64});
  const auto sin = freqs.sin().to(torch::kBFloat16).view({1, T, 1, 64});
  const auto x_in = torch::randn({B, T, 256}, opts).to(torch::kBFloat16);
  const auto ve = torch::randn({B, T, 256}, opts).to(torch::kBFloat16);
  const auto up = torch::randn({B, T, 256}, opts).to(torch::kBFloat16);
  auto run = [&](bool fused) {
    attn->fused = fused;
    attn->zero_grad(true);
    auto x = x_in.clone().requires_grad_();
    const auto y = attn(x, ve, cos, sin, -1);
    (y.to(torch::kFloat32) * up).sum().backward();
    std::vector<torch::Tensor> g{x.grad()};
    for (const auto& p : attn->parameters())
      g.push_back(p.grad().clone());
    return std::pair{y.detach(), g};
  };
  for (const bool fp8 : {false, true}) {
    attn->c_q->fp8 = attn->c_k->fp8 = attn->c_v->fp8 = attn->c_proj->fp8 = fp8;
    const auto [y_f, g_f] = run(true);
    const auto [y_o, g_o] = run(false);
    EXPECT_LT(rel_norm_diff(y_f, y_o), fp8 ? 3e-2 : 1.5e-2) << "fp8=" << fp8;
    for (size_t i = 0; i < g_f.size(); ++i)
      EXPECT_LT(rel_norm_diff(g_f[i], g_o[i]), fp8 ? 6e-2 : 2e-2) << "fp8=" << fp8 << " grad " << i;
  }
}
