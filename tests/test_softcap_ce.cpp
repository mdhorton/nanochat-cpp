// Chunked softcap cross-entropy vs the unchunked ops (as gpt.py): loss and gradients.
#include <gtest/gtest.h>

#include "nanochat/model/gpt.h"
#include "nanochat/model/softcap_ce.h"

using namespace nanochat;
namespace F = torch::nn::functional;

namespace {

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-12)).item<double>();
}

// What GPT::forward does without chunking.
torch::Tensor reference(
      const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& targets, int64_t vocab,
      F::CrossEntropyFuncOptions::reduction_t reduction) {
  auto logits = F::linear(x, w.to(x.scalar_type())).slice(1, 0, vocab).to(torch::kFloat32);
  logits = 15 * torch::tanh(logits / 15);
  return F::cross_entropy(logits, targets, F::CrossEntropyFuncOptions().ignore_index(-1).reduction(reduction));
}

struct Case {
  LossReduction reduction;
  F::CrossEntropyFuncOptions::reduction_t ref;
};

} // namespace

TEST(SoftcapCE, MatchesUnchunked) {
  torch::manual_seed(0);
  const int64_t N = 1000, C = 64, vocab = 1000, padded = 1024; // 1000 rows in chunks of 384: uneven last chunk
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const auto x0 = torch::randn({N, C}, opts).to(torch::kBFloat16);
  const auto w0 = torch::randn({padded, C}, opts) * 0.5;
  auto targets = torch::randint(0, vocab, {N}, opts.dtype(torch::kInt64));
  targets.slice(0, 0, 50).fill_(-1); // ignored rows
  for (const auto& c :
       {Case{LossReduction::Mean, torch::kMean}, Case{LossReduction::Sum, torch::kSum},
        Case{LossReduction::None, torch::kNone}}) {
    auto x1 = x0.clone().requires_grad_(), w1 = w0.clone().requires_grad_();
    auto x2 = x0.clone().requires_grad_(), w2 = w0.clone().requires_grad_();
    auto got = softcap_cross_entropy(x1, w1, targets, vocab, 15, 384, c.reduction);
    auto want = reference(x2, w2, targets, vocab, c.ref);
    EXPECT_LT(rel_diff(got, want), 1e-5);
    const auto upstream = torch::rand_like(want); // exercise a non-trivial incoming gradient
    (got * upstream).sum().backward();
    (want * upstream).sum().backward();
    EXPECT_LT(rel_diff(x1.grad(), x2.grad()), 2e-2);
    EXPECT_LT(rel_diff(w1.grad(), w2.grad()), 2e-2);
    EXPECT_EQ(w1.grad().slice(0, vocab).abs().max().item<float>(), 0.0f); // padded vocab rows get no gradient
  }
}

TEST(SoftcapCE, ModelLossMatchesUnchunked) {
  torch::manual_seed(0);
  GPT model(
        GPTConfig{.sequence_len = 256, .vocab_size = 1000, .n_layer = 4, .n_head = 4, .n_kv_head = 4, .n_embd = 256});
  model->init_weights();
  auto ids = torch::randint(0, 1000, {2, 257}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  auto x = ids.slice(1, 0, -1).contiguous(), y = ids.slice(1, 1).contiguous();
  auto run = [&](int64_t rows) {
    model->set_loss_chunk_rows(rows);
    model->zero_grad(true);
    auto loss = model->forward(x, y);
    loss.backward();
    return std::pair{loss.item<double>(), model->lm_head->weight.grad().clone()};
  };
  const auto [chunked, grad_chunked] = run(100);
  const auto [plain, grad_plain] = run(0);
  EXPECT_NEAR(chunked, plain, 1e-5);
  EXPECT_LT(rel_diff(grad_chunked, grad_plain), 2e-2);
}
