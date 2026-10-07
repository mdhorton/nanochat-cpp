// Fused smear vs gpt.py's op-by-op bf16 path and fp64 references.
#include <gtest/gtest.h>

#include "nanochat/model/ops/smear.h"

using namespace nanochat;

namespace {

namespace F = torch::nn::functional;
using torch::indexing::None;
using torch::indexing::Slice;

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-30)).item<double>();
}

struct Outputs {
  torch::Tensor out, dx, dw, dlambda;
};

// gpt.py's smear on x with w and lambda cast to x's dtype
torch::Tensor smear_ops(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& lambda) {
  const int64_t gate_cols = w.size(1);
  auto lam = lambda.to(x.scalar_type());
  auto gate = lam * torch::sigmoid(
                          F::linear(x.index({Slice(), Slice(1, None), Slice(None, gate_cols)}), w.to(x.scalar_type())));
  auto prev = gate * x.index({Slice(), Slice(None, -1)});
  return torch::cat({x.index({Slice(), Slice(None, 1)}), x.index({Slice(), Slice(1, None)}) + prev}, 1);
}

// fused: smear; else the ops on x in dtype (fp64: the reference). w, lambda: fp32 parameters, as the model's.
Outputs run(
      const torch::Tensor& x_in, const torch::Tensor& w_in, const torch::Tensor& lambda_in, const torch::Tensor& g,
      torch::ScalarType dtype, bool fused) {
  auto x = x_in.to(dtype).clone().requires_grad_();
  auto w = w_in.clone().requires_grad_(), lambda = lambda_in.clone().requires_grad_();
  auto out = fused ? smear(x, w, lambda) : smear_ops(x, w, lambda);
  (out.to(torch::kFloat64) * g).sum().backward();
  return {out.detach(), x.grad(), w.grad(), lambda.grad()};
}

// The kernel's backward formulas in fp64 on the forward's bf16-rounded gate quantities and the bf16 gradient it
// receives (the fused path and the op path both differentiate through those roundings).
Outputs reference(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& lambda, const torch::Tensor& g) {
  const int64_t gate_cols = w.size(1);
  const auto f64 = torch::kFloat64;
  const auto x64 = x.to(f64), g64 = g.to(torch::kBFloat16).to(f64), w64 = w.to(torch::kBFloat16).to(f64);
  const auto lam = lambda.to(torch::kBFloat16).to(f64);
  const auto z = F::linear(x.index({Slice(), Slice(1, None), Slice(None, gate_cols)}), w.to(torch::kBFloat16));
  const auto s = torch::sigmoid(z).to(f64), gate = (lambda.to(torch::kBFloat16) * torch::sigmoid(z)).to(f64);
  const auto dgate = (g64.index({Slice(), Slice(1, None)}) * x64.index({Slice(), Slice(None, -1)})).sum(-1, true);
  const auto dz = dgate * lam * s * (1 - s);
  auto dx = g64.clone();
  dx.index({Slice(), Slice(None, -1)}) += gate * g64.index({Slice(), Slice(1, None)});
  dx.index({Slice(), Slice(1, None), Slice(None, gate_cols)}) += dz * w64;
  return {
        {},
        dx,
        (dz * x64.index({Slice(), Slice(1, None), Slice(None, gate_cols)})).sum({0, 1}, true),
        (dgate * s).sum()};
}

const std::vector<std::vector<int64_t>> kShapes = {{2, 33, 256}, {1, 7, 768}, {4, 64, 768}, {2, 5, 2048}, {3, 2, 512}};

} // namespace

TEST(Smear, MatchesOpByOpAndReference) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  for (const auto& shape : kShapes)
    for (const int64_t gate_cols : {24, 64}) {
      const auto x = torch::randn(shape, opts).to(torch::kBFloat16);
      const auto w = torch::randn({1, gate_cols}, opts) * 0.05;
      const auto lambda = torch::full({1}, 0.3, opts);
      const auto g = torch::randn(shape, opts.dtype(torch::kFloat64));
      const auto f = run(x, w, lambda, g, torch::kBFloat16, true), o = run(x, w, lambda, g, torch::kBFloat16, false);
      const auto n = std::to_string(shape[1]) + "x" + std::to_string(shape[2]) + " gate " + std::to_string(gate_cols);
      // forward: the op path's roundings (the gate logit's sum order could differ, so far it never has)
      EXPECT_TRUE(torch::equal(f.out, o.out)) << n;
      // backward: fp32 inside, rounded once (dx to bf16), summed deterministically
      const auto r = reference(x, w, lambda, g);
      EXPECT_LT(rel_diff(f.dx, r.dx), 4e-3) << n;
      EXPECT_LT(rel_diff(f.dw, r.dw), 1e-4) << n;
      EXPECT_LT(rel_diff(f.dlambda, r.dlambda), 1e-4) << n;
      const auto full = run(x, w, lambda, g, torch::kFloat64, false); // at least as close to fp64 as the op path
      EXPECT_LE(rel_diff(f.dx, full.dx), rel_diff(o.dx, full.dx) * 1.5) << n;
      EXPECT_LE(rel_diff(f.dw, full.dw), rel_diff(o.dw, full.dw) * 1.5) << n;
      const auto f2 = run(x, w, lambda, g, torch::kBFloat16, true);
      EXPECT_TRUE(torch::equal(f.dx, f2.dx) && torch::equal(f.dw, f2.dw) && torch::equal(f.dlambda, f2.dlambda)) << n;
    }
}
