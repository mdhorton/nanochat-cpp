// Fused embedding lookups whose backward accumulates straight into .grad, vs at::embedding + autograd and fp64.
#include <gtest/gtest.h>

#include "nanochat/model/ops/embedding.h"

using namespace nanochat;

namespace {

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-30)).item<double>();
}

} // namespace

TEST(Embedding, AccumulatesIntoGrad) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const int64_t vocab = 1000, B = 4, T = 300, micro_steps = 3;
  const std::vector<int64_t> dims{256, 128, 2048};
  std::vector<torch::Tensor> base;
  for (const auto d : dims)
    base.push_back(torch::randn({vocab, d}, opts).to(torch::kBFloat16));

  std::vector<torch::Tensor> fused, ops, ref;
  for (const auto& b : base) {
    fused.push_back(b.clone().requires_grad_());
    ops.push_back(b.clone().requires_grad_());
    ref.push_back(torch::zeros(b.sizes(), opts.dtype(torch::kFloat64)));
  }
  auto touched = torch::zeros({vocab}, opts.dtype(torch::kBool));
  for (int64_t step = 0; step < micro_steps; ++step) {
    // a hot id (runs far longer than a block) and the ends of the vocab
    auto idx = torch::randint(1, vocab - 1, {B, T}, opts.dtype(torch::kInt64));
    idx.masked_fill_(torch::rand({B, T}, opts) < 0.3, 7);
    idx.index_put_({0, 0}, 0);
    idx.index_put_({1, 5}, vocab - 1);
    touched.index_fill_(0, idx.flatten(), true);

    const auto out = embeddings(idx, fused);
    auto loss_f = torch::zeros({}, opts.dtype(torch::kFloat32));
    auto loss_o = torch::zeros({}, opts.dtype(torch::kFloat32));
    for (size_t i = 0; i < dims.size(); ++i) {
      const auto up = torch::randn({B, T, dims[i]}, opts).to(torch::kBFloat16);
      const auto o = at::embedding(ops[i], idx);
      EXPECT_TRUE(torch::equal(out[i], o));
      loss_f = loss_f + (out[i].to(torch::kFloat32) * up).sum();
      loss_o = loss_o + (o.to(torch::kFloat32) * up).sum();
      ref[i].index_add_(0, idx.flatten(), up.to(torch::kFloat64).view({-1, dims[i]}));
    }
    loss_f.backward();
    loss_o.backward();
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    const auto& g = fused[i].grad();
    ASSERT_TRUE(g.defined());
    const auto d = std::to_string(dims[i]);
    // untouched rows stay exactly zero; touched rows are at least as close to fp64 as autograd's
    EXPECT_EQ(g.index({touched.logical_not()}).abs().sum().item<double>(), 0.0) << d;
    const double err_f = rel_diff(g, ref[i]), err_o = rel_diff(ops[i].grad(), ref[i]);
    EXPECT_LT(err_f, 1e-2) << d; // bf16 spacing: 3 accumulations, each rounded once
    EXPECT_LE(err_f, err_o) << d;
  }

  // deterministic
  std::vector<torch::Tensor> again;
  for (const auto& b : base)
    again.push_back(b.clone().requires_grad_());
  auto idx = torch::randint(0, vocab, {B, T}, opts.dtype(torch::kInt64));
  idx.masked_fill_(torch::rand({B, T}, opts) < 0.3, 7);
  std::vector<torch::Tensor> first;
  for (int rep = 0; rep < 2; ++rep) {
    for (auto& w : again)
      w.mutable_grad() = torch::Tensor();
    const auto out = embeddings(idx, again);
    auto loss = torch::zeros({}, opts.dtype(torch::kFloat32));
    torch::manual_seed(1);
    for (size_t i = 0; i < dims.size(); ++i)
      loss = loss + (out[i].to(torch::kFloat32) * torch::randn({B, T, dims[i]}, opts)).sum();
    loss.backward();
    for (size_t i = 0; i < dims.size(); ++i)
      if (rep == 0)
        first.push_back(again[i].grad().clone());
      else
        EXPECT_TRUE(torch::equal(first[i], again[i].grad())) << dims[i];
  }
}
