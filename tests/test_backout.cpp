// Fused backout vs gpt.py's ops (bit-identical but for dlambda, checked against an fp64 sum).
#include <gtest/gtest.h>

#include "nanochat/model/ops/backout.h"

using namespace nanochat;

namespace {

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-30)).item<double>();
}

struct Outputs {
  torch::Tensor out, dx, dxb, dlambda;
};

// fused: backout; else the ops on x, xb in dtype; lambda: an fp32 parameter, as the model's
Outputs run(
      const torch::Tensor& x_in, const torch::Tensor& xb_in, const torch::Tensor& lambda_in, const torch::Tensor& g,
      torch::ScalarType dtype, bool fused) {
  auto x = x_in.to(dtype).clone().requires_grad_(), xb = xb_in.to(dtype).clone().requires_grad_();
  auto lambda = lambda_in.clone().requires_grad_();
  auto out = fused ? backout(x, xb, lambda) : x - lambda.to(dtype) * xb;
  (out.to(torch::kFloat64) * g).sum().backward();
  return {out.detach(), x.grad(), xb.grad(), lambda.grad()};
}

} // namespace

TEST(Backout, MatchesOpByOpAndReference) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  for (const auto& shape : std::vector<std::vector<int64_t>>{{2, 33, 256}, {1, 7, 768}, {3, 64, 768}, {2, 3, 2048}}) {
    const auto x = torch::randn(shape, opts).to(torch::kBFloat16), xb = torch::randn(shape, opts).to(torch::kBFloat16);
    const auto lambda = torch::full({1}, 0.2, opts);
    const auto g = torch::randn(shape, opts.dtype(torch::kFloat64));
    const auto f = run(x, xb, lambda, g, torch::kBFloat16, true), o = run(x, xb, lambda, g, torch::kBFloat16, false);
    const auto n = std::to_string(shape[1]) + "x" + std::to_string(shape[2]);
    EXPECT_TRUE(torch::equal(f.out, o.out)) << n;
    EXPECT_TRUE(torch::equal(f.dx, o.dx)) << n;
    EXPECT_TRUE(torch::equal(f.dxb, o.dxb)) << n;
    // fp32 sum of the bf16 gradient's products (the ops round each product to bf16)
    const auto dlambda_ref = -(g.to(torch::kBFloat16).to(torch::kFloat64) * xb.to(torch::kFloat64)).sum();
    EXPECT_LT(rel_diff(f.dlambda, dlambda_ref), 1e-3) << n;
    const auto f2 = run(x, xb, lambda, g, torch::kBFloat16, true);
    EXPECT_TRUE(torch::equal(f.dlambda, f2.dlambda)) << n; // deterministic
  }
}
