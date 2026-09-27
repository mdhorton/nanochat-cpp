// Fused resid/x0 lambda blend vs an fp64 reference and vs gpt.py's op-by-op bf16 path: outputs and gradients.
#include <gtest/gtest.h>

#include "nanochat/model/gpt.h"
#include "nanochat/model/lambda_blend.h"

using namespace nanochat;

namespace {

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-30)).item<double>();
}

struct Grads {
  torch::Tensor out, dx, dx0, dlr, dl0;
};

// fused: lambda_blend; else gpt.py's ops in x's dtype
Grads run(
      const torch::Tensor& x_in, const torch::Tensor& x0_in, const torch::Tensor& lr_in, const torch::Tensor& l0_in,
      const torch::Tensor& upstream, int64_t layer, bool fused) {
  auto x = x_in.clone().requires_grad_(), x0 = x0_in.clone().requires_grad_();
  auto lr = lr_in.clone().requires_grad_(), l0 = l0_in.clone().requires_grad_();
  auto out = fused ? lambda_blend(x, x0, lr, l0, layer) : lr[layer] * x + l0[layer] * x0;
  (out.to(torch::kFloat64) * upstream).sum().backward();
  return {out.detach(), x.grad(), x0.grad(), lr.grad(), l0.grad()};
}

} // namespace

TEST(LambdaBlend, MatchesReference) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  for (const auto& shape :
       {std::vector<int64_t>{3, 5, 7}, std::vector<int64_t>{4, 64, 96},
        std::vector<int64_t>{8, 2048, 768}}) { // odd size: tail path
    const int64_t n_layer = 5, layer = 3;
    const auto x = torch::randn(shape, opts).to(torch::kBFloat16), x0 = torch::randn(shape, opts).to(torch::kBFloat16);
    const auto lr = torch::rand({n_layer}, opts) + 0.5, l0 = torch::rand({n_layer}, opts) * 0.3;
    const auto upstream = torch::randn(shape, opts.dtype(torch::kFloat64)).to(torch::kBFloat16).to(torch::kFloat64);

    const auto f = run(x, x0, lr, l0, upstream, layer, true);
    const auto o = run(x, x0, lr, l0, upstream, layer, false);
    const auto r = run(
          x.to(torch::kFloat64), x0.to(torch::kFloat64), lr.to(torch::kFloat64), l0.to(torch::kFloat64), upstream,
          layer, false);
    const auto n = std::to_string(x.numel());
    // bit-identical to the op-by-op path (full-precision math shifts training, see the kernel)
    EXPECT_TRUE(torch::equal(f.out, o.out)) << n;
    EXPECT_TRUE(torch::equal(f.dx, o.dx)) << n;
    EXPECT_TRUE(torch::equal(f.dx0, o.dx0)) << n;
    // lambda grads summed in fp32 (Python rounds g * x to bf16 first)
    EXPECT_LT(rel_diff(f.dlr, r.dlr), 1e-5) << n;
    EXPECT_LT(rel_diff(f.dl0, r.dl0), 1e-5) << n;
    EXPECT_EQ(f.dlr.abs().sum().item<float>(), f.dlr[layer].abs().item<float>()) << n; // other layers: 0
    EXPECT_EQ(f.dl0.abs().sum().item<float>(), f.dl0[layer].abs().item<float>()) << n;
    // deterministic
    const auto f2 = run(x, x0, lr, l0, upstream, layer, true);
    EXPECT_TRUE(torch::equal(f.dlr, f2.dlr) && torch::equal(f.dl0, f2.dl0)) << n;
  }
}

TEST(LambdaBlend, ModelMatchesOpByOp) {
  torch::manual_seed(0);
  GPT model(
        GPTConfig{.sequence_len = 256, .vocab_size = 1000, .n_layer = 4, .n_head = 4, .n_kv_head = 4, .n_embd = 256});
  model->init_weights();
  {
    torch::NoGradGuard no_grad;
    for (auto& p : model->parameters()) // c_proj starts at zero, which would make the blocks identities
      p.add_(torch::randn_like(p, torch::kFloat32).to(p.scalar_type()) * 0.02);
  }
  auto ids = torch::randint(0, 1000, {2, 257}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  auto x = ids.slice(1, 0, -1).contiguous(), y = ids.slice(1, 1).contiguous();
  model->set_attention(Attention::SDPA); // deterministic
  auto step = [&](bool fused) {
    model->set_fused(fused);
    for (const auto& m : *model->transformer->h) { // blend only
      m->as<BlockImpl>()->attn->fused = false;
      m->as<BlockImpl>()->mlp->fused = false;
    }
    model->zero_grad(true);
    auto loss = model->forward(x, y);
    loss.backward();
    return std::tuple{
          loss.item<double>(), model->resid_lambdas.grad().clone(), model->x0_lambdas.grad().clone(),
          model->transformer->wte->weight.grad().clone()};
  };
  const auto [loss_f, gr_f, g0_f, gw_f] = step(true);
  const auto [loss_o, gr_o, g0_o, gw_o] = step(false);
  // forward, dx and dx0 are bit-identical; x0's many bf16 grads may be summed in another order
  EXPECT_EQ(loss_f, loss_o);
  EXPECT_LT(rel_diff(gw_f, gw_o), 3e-2);
  // lambda grads: summed in fp32 vs bf16 products summed to a bf16 result
  EXPECT_LT(rel_diff(gr_f, gr_o), 2e-2);
  EXPECT_LT(rel_diff(g0_f, g0_o), 2e-2);
}
