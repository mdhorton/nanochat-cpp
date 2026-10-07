// Fused rotary + QK norm vs an fp64 reference and vs gpt.py's op-by-op bf16 path: outputs and gradients.
#include <gtest/gtest.h>

#include "nanochat/model/attention/rotary_norm.h"
#include "nanochat/model/gpt.h"

using namespace nanochat;
using torch::indexing::None;
using torch::indexing::Slice;

namespace {

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-30)).item<double>();
}

// (1, seq_len, 1, D / 2) bf16, as GPT::precompute_rotary
std::pair<torch::Tensor, torch::Tensor> rotary_cache(int64_t seq_len, int64_t D) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto inv_freq = torch::pow(100000.0, torch::arange(0, D, 2, opts) / D).reciprocal();
  auto freqs = torch::outer(torch::arange(seq_len, opts), inv_freq);
  return {
        freqs.cos().to(torch::kBFloat16).unsqueeze(0).unsqueeze(2),
        freqs.sin().to(torch::kBFloat16).unsqueeze(0).unsqueeze(2)};
}

// gpt.py: apply_rotary_emb, then norm(x) * scale, in x's dtype
torch::Tensor reference(const torch::Tensor& x, const torch::Tensor& cos_full, const torch::Tensor& sin_full) {
  const int64_t T = x.size(1), d = x.size(3) / 2;
  const auto cos = cos_full.index({Slice(), Slice(None, T)}).to(x.scalar_type());
  const auto sin = sin_full.index({Slice(), Slice(None, T)}).to(x.scalar_type());
  auto x1 = x.index({"...", Slice(None, d)}), x2 = x.index({"...", Slice(d, None)});
  auto y = torch::cat({x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos}, 3);
  return at::rms_norm(
               y, {y.size(-1)}, {},
               x.scalar_type() == torch::kFloat64 ? 1.1920928955078125e-07 : std::optional<double>()) *
         1.2;
}

struct Shape {
  int64_t B, T, H, D, cache_len;
};

} // namespace

TEST(RotaryNorm, MatchesReference) {
  torch::manual_seed(0);
  for (const auto& s : {Shape{2, 37, 3, 64, 100}, Shape{1, 16, 2, 128, 16}, Shape{3, 9, 6, 256, 20}}) {
    const auto [cos, sin] = rotary_cache(s.cache_len, s.D);
    auto x0 = torch::randn({s.B, s.T, s.H, s.D}, torch::TensorOptions().device(torch::kCUDA)).to(torch::kBFloat16);
    x0.index_put_({0, 0}, x0.index({0, 0}) * 1e-4); // tiny rows: eps matters
    const auto upstream = torch::randn_like(x0, torch::kFloat32).to(torch::kBFloat16);

    auto xf = x0.clone().requires_grad_(), xb = x0.clone().requires_grad_();
    auto x64 = x0.to(torch::kFloat64).requires_grad_();
    auto fused = rotary_rms_norm(xf, cos, sin, 1.2);
    auto ops = reference(xb, cos, sin); // gpt.py in bf16
    auto want = reference(x64, cos, sin);
    (fused.to(torch::kFloat32) * upstream).sum().backward();
    (ops.to(torch::kFloat32) * upstream).sum().backward();
    (want * upstream.to(torch::kFloat64)).sum().backward();

    // fp32 inside with one rounding: within bf16 resolution of fp64, and no worse than the op-by-op path
    const double fused_err = rel_diff(fused, want), ops_err = rel_diff(ops, want);
    EXPECT_LT(fused_err, 8e-3) << "D=" << s.D;
    EXPECT_LE(fused_err, ops_err) << "D=" << s.D;
    const double fused_gerr = rel_diff(xf.grad(), x64.grad()), ops_gerr = rel_diff(xb.grad(), x64.grad());
    EXPECT_LT(fused_gerr, 8e-3) << "D=" << s.D;
    EXPECT_LE(fused_gerr, ops_gerr * 1.5) << "D=" << s.D;
    // tiny rows (eps = float epsilon, as F.rms_norm): same as the op-by-op path
    EXPECT_LT(rel_diff(fused.index({0, 0}), ops.index({0, 0})), 2e-2) << "D=" << s.D;
  }
}

TEST(RotaryNorm, ModelMatchesOpByOp) {
  torch::manual_seed(0);
  GPT model(
        GPTConfig{.sequence_len = 256, .vocab_size = 1000, .n_layer = 4, .n_head = 4, .n_kv_head = 2, .n_embd = 256});
  model->init_weights();
  {
    torch::NoGradGuard no_grad;
    for (auto& p : model->parameters()) // c_proj starts at zero, which would leave q/k without gradient
      p.add_(torch::randn_like(p, torch::kFloat32).to(p.scalar_type()) * 0.02);
  }
  auto ids = torch::randint(0, 1000, {2, 257}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  auto x = ids.slice(1, 0, -1).contiguous(), y = ids.slice(1, 1).contiguous();
  const auto& attn = model->transformer->h[1]->as<BlockImpl>()->attn;
  auto run = [&](bool fused) {
    model->set_fused(fused);
    model->zero_grad(true);
    auto loss = model->forward(x, y);
    loss.backward();
    return std::tuple{loss.item<double>(), attn->c_q->weight.grad().clone(), attn->c_k->weight.grad().clone()};
  };
  const auto [loss_f, gq_f, gk_f] = run(true);
  const auto [loss_o, gq_o, gk_o] = run(false);
  EXPECT_NEAR(loss_f, loss_o, 1e-3);
  EXPECT_LT(rel_diff(gq_f, gq_o), 3e-2);
  EXPECT_LT(rel_diff(gk_f, gk_o), 3e-2);
}
