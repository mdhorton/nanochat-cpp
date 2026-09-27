// Fused relu^2 (and its FP8 MLP) vs gpt.py's op-by-op path: bit-identical outputs and gradients.
#include <gtest/gtest.h>

#include "nanochat/model/gpt.h"
#include "nanochat/model/relu_square.h"

using namespace nanochat;

TEST(ReluSquare, MatchesOpByOp) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  for (const auto& shape : {std::vector<int64_t>{3, 5, 7}, std::vector<int64_t>{8, 2048, 768}}) { // odd: tail path
    auto h_in = (torch::randn(shape, opts) * 4).to(torch::kBFloat16);
    h_in.view(-1).slice(0, 0, 10).zero_(); // relu's kink
    const auto upstream = torch::randn(shape, opts).to(torch::kBFloat16);
    auto run = [&](bool fused) {
      auto h = h_in.clone().requires_grad_();
      auto a = fused ? relu_square(h) : torch::relu(h).square();
      a.backward(upstream);
      return std::pair{a.detach(), h.grad()};
    };
    const auto [a_f, g_f] = run(true);
    const auto [a_o, g_o] = run(false);
    const auto n = std::to_string(h_in.numel());
    EXPECT_TRUE(torch::equal(a_f, a_o)) << n;
    EXPECT_TRUE(torch::equal(g_f, g_o)) << n;
  }
}

// With FP8, the fused MLP also folds relu^2 into the quantization of c_proj's input and of dh.
TEST(ReluSquare, MlpMatchesOpByOp) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const GPTConfig config{.n_embd = 256};
  MLP mlp(config, opts.dtype(torch::kFloat32));
  {
    torch::NoGradGuard no_grad;
    for (auto& p : mlp->parameters())
      p.normal_(0, 0.05);
  }
  const auto x_in = torch::randn({3, 208, config.n_embd}, opts).to(torch::kBFloat16); // 624 rows: FP8 needs % 16
  const auto upstream = torch::randn({3, 208, config.n_embd}, opts).to(torch::kBFloat16);
  for (const bool fp8 : {false, true}) {
    mlp->c_fc->fp8 = mlp->c_proj->fp8 = fp8;
    auto run = [&](bool fused) {
      mlp->fused = fused;
      mlp->zero_grad(true);
      auto x = x_in.clone().requires_grad_();
      auto out = mlp(x);
      out.backward(upstream);
      return std::tuple{out.detach(), x.grad(), mlp->c_fc->weight.grad().clone(), mlp->c_proj->weight.grad().clone()};
    };
    const auto [out_f, dx_f, dfc_f, dproj_f] = run(true);
    const auto [out_o, dx_o, dfc_o, dproj_o] = run(false);
    EXPECT_TRUE(torch::equal(out_f, out_o)) << "fp8=" << fp8;
    EXPECT_TRUE(torch::equal(dx_f, dx_o)) << "fp8=" << fp8;
    EXPECT_TRUE(torch::equal(dfc_f, dfc_o)) << "fp8=" << fp8;
    EXPECT_TRUE(torch::equal(dproj_f, dproj_o)) << "fp8=" << fp8;
  }
}
