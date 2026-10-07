// Fused residual add + lambda blend + rms_norm vs gpt.py's op-by-op bf16 path and an fp64 reference.
#include <gtest/gtest.h>

#include "nanochat/model/gpt.h"
#include "nanochat/model/ops/residual_norm.h"

using namespace nanochat;

namespace {

constexpr double kEps = 1.1920928955078125e-07; // F.rms_norm's default for bf16 (float epsilon)

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-30)).item<double>();
}

struct Inputs {
  torch::Tensor x, r, x0, lr, l0, g_res, g_n; // r, x0, g_res, g_n may be undefined
  int64_t layer = 3;
};

struct Outputs {
  torch::Tensor res, n, dx, dr, dx0, dlr, dl0;
};

torch::Tensor grad_or_empty(const torch::Tensor& t) {
  return t.defined() ? t.grad() : torch::Tensor();
}

// fused: residual_norm; else gpt.py's ops in the inputs' dtype (fp64: the reference)
Outputs run(const Inputs& in, torch::ScalarType dtype, bool fused) {
  auto leaf = [&](const torch::Tensor& t, bool cast = true) {
    return t.defined() ? (cast ? t.to(dtype) : t).clone().requires_grad_() : torch::Tensor();
  };
  auto x = leaf(in.x), r = leaf(in.r), x0 = leaf(in.x0);
  const bool f64 = dtype == torch::kFloat64;
  auto lr = leaf(in.lr, f64), l0 = leaf(in.l0, f64);
  torch::Tensor res, n;
  if (fused)
    std::tie(res, n) = residual_norm(x, r, x0, lr, l0, in.layer);
  else {
    auto s = r.defined() ? x + r : x;
    if (x0.defined()) {
      auto scaled = lr[in.layer] * s;
      res = scaled + l0[in.layer] * x0;
    }
    else
      res = s;
    n = at::rms_norm(res, {res.size(-1)}, {}, f64 ? kEps : std::optional<double>());
  }
  auto loss = torch::zeros({}, in.x.options().dtype(torch::kFloat64));
  if (in.g_res.defined())
    loss = loss + (res.to(torch::kFloat64) * in.g_res).sum();
  if (in.g_n.defined())
    loss = loss + (n.to(torch::kFloat64) * in.g_n).sum();
  loss.backward();
  return {res.detach(),      n.detach(),        x.grad(),         grad_or_empty(r),
          grad_or_empty(x0), grad_or_empty(lr), grad_or_empty(l0)};
}

// bf16 values as fp64
torch::Tensor randn_bf16(const std::vector<int64_t>& shape, double scale = 1) {
  return (torch::randn(shape, torch::TensorOptions().device(torch::kCUDA)) * scale)
        .to(torch::kBFloat16)
        .to(torch::kFloat64);
}

Inputs make_inputs(const std::vector<int64_t>& shape, bool add, bool blend, bool g_res, bool g_n) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  Inputs in;
  in.x = randn_bf16(shape).to(torch::kBFloat16);
  in.x.view(-1).slice(0, 0, shape.back()).mul_(1e-4); // a tiny row: eps matters
  if (add)
    in.r = randn_bf16(shape, 0.5).to(torch::kBFloat16);
  if (blend) {
    in.x0 = randn_bf16(shape).to(torch::kBFloat16);
    in.lr = torch::rand({5}, opts) + 0.5;
    in.l0 = torch::rand({5}, opts) * 0.3;
  }
  if (g_res)
    in.g_res = randn_bf16(shape);
  if (g_n)
    in.g_n = randn_bf16(shape);
  return in;
}

const std::vector<std::vector<int64_t>> kShapes =
      {{3, 24}, {5, 136}, {4, 64, 768}, {2, 33, 1536}, {1, 7, 2048}}; // partial and full warp passes

std::string name(const std::vector<int64_t>& shape, bool add, bool blend) {
  return std::to_string(shape.back()) + (add ? " add" : "") + (blend ? " blend" : "");
}

} // namespace

// Without a norm gradient, everything but the lambda grads is bit-identical to the op path.
TEST(ResidualNorm, AddBlendMatchOpByOp) {
  torch::manual_seed(0);
  for (const auto& shape : kShapes)
    for (const auto& [add, blend] : {std::pair{true, false}, {false, true}, {true, true}}) {
      const auto in = make_inputs(shape, add, blend, true, false);
      const auto f = run(in, torch::kBFloat16, true), o = run(in, torch::kBFloat16, false);
      const auto n = name(shape, add, blend);
      EXPECT_TRUE(torch::equal(f.res, o.res)) << n;
      EXPECT_TRUE(torch::equal(f.dx, o.dx)) << n;
      if (add)
        EXPECT_TRUE(torch::equal(f.dr, o.dr)) << n;
      if (!blend)
        continue;
      EXPECT_TRUE(torch::equal(f.dx0, o.dx0)) << n;
      // summed in fp32 (the op path rounds g * s to bf16 first); fp64 reference on the rounded s = x + r
      auto in_s = in;
      if (add) {
        in_s.x = in.x + in.r;
        in_s.r = torch::Tensor();
      }
      const auto r = run(in_s, torch::kFloat64, false);
      EXPECT_LT(rel_diff(f.dlr, r.dlr), 1e-5) << n;
      EXPECT_LT(rel_diff(f.dl0, r.dl0), 1e-5) << n;
      EXPECT_EQ(f.dlr.abs().sum().item<float>(), f.dlr[in.layer].abs().item<float>()) << n; // other layers: 0
      EXPECT_EQ(f.dl0.abs().sum().item<float>(), f.dl0[in.layer].abs().item<float>()) << n;
      const auto f2 = run(in, torch::kBFloat16, true);
      EXPECT_TRUE(torch::equal(f.dlr, f2.dlr) && torch::equal(f.dl0, f2.dl0)) << n; // deterministic
    }
}

// The norm (fp32 inside) is at least as close to fp64 as torch's bf16 rms_norm, forward and backward.
TEST(ResidualNorm, NormMatchesReference) {
  torch::manual_seed(0);
  double lambda_err_f = 0, lambda_err_o = 0;
  for (const auto& shape : kShapes)
    for (const auto& [add, blend] : {std::pair{false, false}, {true, false}, {true, true}}) {
      const auto in = make_inputs(shape, add, blend, true, true);
      const auto f = run(in, torch::kBFloat16, true), o = run(in, torch::kBFloat16, false);
      const auto r = run(in, torch::kFloat64, false);
      const auto n = name(shape, add, blend);
      EXPECT_LT(rel_diff(f.n, r.n), 8e-3) << n;
      EXPECT_LE(rel_diff(f.n, r.n), rel_diff(o.n, r.n) * 1.5) << n;
      EXPECT_LT(rel_diff(f.dx, r.dx), 1e-2) << n;
      EXPECT_LE(rel_diff(f.dx, r.dx), rel_diff(o.dx, r.dx) * 1.5) << n;
      if (blend) {
        EXPECT_LT(rel_diff(f.dx0, r.dx0), 1e-2) << n;
        EXPECT_LE(rel_diff(f.dx0, r.dx0), rel_diff(o.dx0, r.dx0) * 1.5) << n;
        lambda_err_f += rel_diff(f.dlr, r.dlr) + rel_diff(f.dl0, r.dl0);
        lambda_err_o += rel_diff(o.dlr, r.dlr) + rel_diff(o.dl0, r.dl0);
      }
    }
  // lambda grads cancel heavily, so single cases swing either way (up to ~10% off fp64 for both paths)
  EXPECT_LE(lambda_err_f, lambda_err_o * 1.5);
}

// residual_norm_mx: quantize_mx(n) bit for bit (both layouts), n's gate columns, the same res and gradients.
TEST(ResidualNorm, MxMatchesQuantize) {
  torch::manual_seed(0);
  const auto u8 = [](const torch::Tensor& t) {
    return t.view(torch::kUInt8);
  };
  const auto leaf = [](const torch::Tensor& t) {
    return t.defined() ? t.clone().requires_grad_() : torch::Tensor();
  };
  for (const auto& shape : std::vector<std::vector<int64_t>>{{256, 768}, {2, 128, 1536}, {384, 256}})
    for (const auto& [add, blend] : {std::pair{false, false}, {true, false}, {true, true}})
      for (const int64_t gate_cols : {0, 12}) {
        const auto in = make_inputs(shape, add, blend, true, true);
        const int64_t C = shape.back(), N = in.x.numel() / C;
        ASSERT_TRUE(residual_norm_mx_fits(N, C));
        const auto what = name(shape, add, blend) + " gate " + std::to_string(gate_cols);
        auto x = leaf(in.x), r = leaf(in.r), x0 = leaf(in.x0), lr = leaf(in.lr), l0 = leaf(in.l0);
        auto [res, n] = residual_norm(x, r, x0, lr, l0, in.layer);
        const auto q = quantize_mx(n.detach().reshape({N, C}));
        auto xm = leaf(in.x), rm = leaf(in.r), x0m = leaf(in.x0), lrm = leaf(in.lr), l0m = leaf(in.l0);
        const auto o = residual_norm_mx(xm, rm, x0m, lrm, l0m, in.layer, {}, gate_cols);
        EXPECT_TRUE(torch::equal(o.res, res)) << what;
        EXPECT_TRUE(torch::equal(u8(o.n_mx.data), u8(q.data))) << what;
        EXPECT_TRUE(torch::equal(u8(o.n_mx.inv_scale), u8(q.inv_scale))) << what;
        EXPECT_TRUE(torch::equal(u8(o.n_mx.data_t), u8(q.data_t))) << what;
        EXPECT_TRUE(torch::equal(u8(o.n_mx.inv_scale_t), u8(q.inv_scale_t))) << what;
        if (gate_cols > 0)
          EXPECT_TRUE(torch::equal(o.n.narrow(-1, 0, gate_cols), n.narrow(-1, 0, gate_cols))) << what;
        const auto g_res = in.g_res.to(torch::kBFloat16), g_n = in.g_n.to(torch::kBFloat16);
        torch::autograd::backward({res, n}, {g_res, g_n});
        torch::autograd::backward({o.res, o.n}, {g_res, g_n});
        EXPECT_TRUE(torch::equal(xm.grad(), x.grad())) << what;
        if (add)
          EXPECT_TRUE(torch::equal(rm.grad(), r.grad())) << what;
        if (blend) {
          EXPECT_TRUE(torch::equal(x0m.grad(), x0.grad())) << what;
          EXPECT_TRUE(torch::equal(lrm.grad(), lr.grad()) && torch::equal(l0m.grad(), l0.grad())) << what;
        }
      }
}

TEST(ResidualNorm, ModelMatchesOpByOp) {
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
    for (const auto& m : *model->transformer->h) { // residual path only
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
  EXPECT_NEAR(loss_f, loss_o, 1e-3);
  EXPECT_LT(rel_diff(gw_f, gw_o), 3e-2);
  // the norms' rounding differences reach layer 0's lambda grads, which cancel heavily (~6% here; kernel-level
  // accuracy is checked above)
  EXPECT_LT(rel_diff(gr_f, gr_o), 1e-1);
  EXPECT_LT(rel_diff(g0_f, g0_o), 1e-1);
}

// X0Grad sums x0's gradient over the layers in one buffer: bit-identical to autograd's adds, also over a retained
// graph's second backward.
TEST(ResidualNorm, X0GradMatchesAutograd) {
  torch::manual_seed(0);
  const std::vector<int64_t> shape{4, 64, 768};
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const auto x0_in = randn_bf16(shape).to(torch::kBFloat16), w = randn_bf16(shape, 0.5).to(torch::kBFloat16);
  const auto lr_in = torch::rand({4}, opts) + 0.5, l0_in = torch::rand({4}, opts) * 0.3;
  const auto g = randn_bf16(shape);
  for (const bool x_is_x0 : {true, false}) {
    auto run_chain = [&](bool sum) {
      auto x0 = x0_in.clone().requires_grad_(), lr = lr_in.clone().requires_grad_(),
           l0 = l0_in.clone().requires_grad_();
      const auto x0_grad = sum ? c10::make_intrusive<X0Grad>() : c10::intrusive_ptr<X0Grad>();
      auto x = x_is_x0 ? x0 : x0 * 2;
      torch::Tensor pending;
      for (int64_t i = 0; i < 4; ++i) {
        auto [res, n] = residual_norm(x, pending, x0, lr, l0, i, x0_grad);
        x = res + n * w;
        pending = i % 2 == 0 ? n * w : torch::Tensor();
      }
      auto loss = (x.to(torch::kFloat64) * g).sum();
      loss.backward({}, true);
      loss.backward();
      return std::tuple{x0.grad(), lr.grad(), l0.grad()};
    };
    const auto [dx0_s, dlr_s, dl0_s] = run_chain(true);
    const auto [dx0_a, dlr_a, dl0_a] = run_chain(false);
    EXPECT_TRUE(torch::equal(dx0_s, dx0_a)) << x_is_x0;
    EXPECT_TRUE(torch::equal(dlr_s, dlr_a) && torch::equal(dl0_s, dl0_a)) << x_is_x0;
  }
}
