// Chunked softcap cross-entropy vs the unchunked ops (as gpt.py): loss and gradients.
#include <gtest/gtest.h>

#include "nanochat/model/fp8.h"
#include "nanochat/model/gpt.h"
#include "nanochat/model/softcap_ce.h"
#include "nanochat/model/softcap_ce_kernel.h"

#include <ATen/cuda/CUDAContext.h>

using namespace nanochat;
namespace F = torch::nn::functional;

namespace {

double rel_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).abs().max() / (b64.abs().max() + 1e-12)).item<double>();
}

double rel_norm_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).norm() / (b64.norm() + 1e-30)).item<double>();
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

// FP8 lm_head: chunked vs the unchunked Float8Matmul path (gpt.py with fp8). Tensorwise: one chunk shares python's
// gradient scale, several chunks scale each chunk's gradient on its own. MX: block scales, chunking changes none.
TEST(SoftcapCE, Fp8MatchesUnchunked) {
  torch::manual_seed(0);
  const int64_t N = 1024, C = 128, vocab = 1000, padded = 1024;
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const auto x0 = torch::randn({N, C}, opts).to(torch::kBFloat16);
  const auto w0 = torch::randn({padded, C}, opts) * 0.5;
  auto targets = torch::randint(0, vocab, {N}, opts.dtype(torch::kInt64));
  targets.slice(0, 0, 48).fill_(-1);
  for (const auto recipe : {Fp8Recipe::Tensorwise, Fp8Recipe::Mx}) {
    auto x2 = x0.clone().requires_grad_(), w2 = w0.clone().requires_grad_();
    auto logits = fp8_matmul(x2, w2, nullptr, recipe).slice(1, 0, vocab).to(torch::kFloat32);
    logits = 15 * torch::tanh(logits / 15);
    const auto want = F::cross_entropy(logits, targets, F::CrossEntropyFuncOptions().ignore_index(-1));
    want.backward();
    for (const int64_t chunk : {N, N / 4}) {
      auto x1 = x0.clone().requires_grad_(), w1 = w0.clone().requires_grad_();
      const auto got = softcap_cross_entropy(
            x1, w1, targets, vocab, 15, chunk, LossReduction::Mean, true, nullptr, recipe);
      got.backward();
      const auto what = std::to_string(chunk) + (recipe == Fp8Recipe::Mx ? " mx" : "");
      EXPECT_LT(rel_diff(got, want), 1e-5) << what;
      EXPECT_LT(rel_norm_diff(x1.grad(), x2.grad()), 1e-3) << what;
      // Float8Matmul rounds grad_w to bf16; the chunks accumulate it in fp32
      EXPECT_LT(rel_norm_diff(w1.grad(), w2.grad()), 5e-3) << what;
      EXPECT_EQ(w1.grad().slice(0, vocab).abs().max().item<float>(), 0.0f) << what;
    }
  }
}

// The MX gradient kernel (from the logits and lse) vs the bf16 gradient kernel then quantize_mx: bit-identical.
TEST(SoftcapCE, MxGradMatchesQuantizedGrad) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const int64_t rows = 256, padded = 512, vocab = 500;
  const auto logits = (torch::randn({rows, padded}, opts) * 8).to(torch::kBFloat16);
  auto targets = torch::randint(vocab, {rows}, opts.dtype(torch::kInt64));
  targets.slice(0, 0, 7).fill_(-1); // ignored rows
  const auto grad_scale = torch::rand({rows}, opts) + 0.5;
  const auto num_valid = (targets >= 0).sum();
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  for (const bool per_row : {false, true}) {
    const float* scale = per_row ? grad_scale.data_ptr<float>() : nullptr;
    const int64_t* valid = per_row ? nullptr : num_valid.data_ptr<int64_t>();
    auto grad = logits.clone();
    auto loss_ref = torch::empty({rows}, opts), loss = torch::empty({rows}, opts), lse = torch::empty({rows}, opts);
    kernels::softcap_ce(
          grad.data_ptr(), rows, padded, targets.data_ptr<int64_t>(), vocab, padded, 15.f, loss_ref.data_ptr<float>(),
          scale, 1, valid, true, nullptr, nullptr, stream);
    const auto ref = quantize_mx(grad);
    kernels::softcap_ce(
          logits.data_ptr(), rows, padded, targets.data_ptr<int64_t>(), vocab, padded, 15.f, loss.data_ptr<float>(),
          scale, 1, valid, false, nullptr, lse.data_ptr<float>(), stream);
    const auto q = empty_mx(rows, padded, opts);
    const auto [out, out_t] = mx_outs(q);
    kernels::softcap_ce_grad_mx(
          logits.data_ptr(), rows, padded, targets.data_ptr<int64_t>(), lse.data_ptr<float>(), vocab, padded, 15.f,
          scale, 1, valid, out, out_t, stream);
    const auto bits = [](const torch::Tensor& t) {
      return t.view(torch::kUInt8);
    };
    EXPECT_TRUE(torch::equal(loss, loss_ref)) << per_row;
    EXPECT_TRUE(torch::equal(bits(q.data), bits(ref.data))) << per_row;
    EXPECT_TRUE(torch::equal(bits(q.data_t), bits(ref.data_t))) << per_row;
    EXPECT_TRUE(torch::equal(bits(q.inv_scale), bits(ref.inv_scale))) << per_row;
    EXPECT_TRUE(torch::equal(bits(q.inv_scale_t), bits(ref.inv_scale_t))) << per_row;
  }
}

// Micro-steps: backward adds each loss's weight gradient into .grad, a leaf weight's directly, a non-leaf's through
// autograd. MX: every chunk's transposed gradient feeds one grad_w GEMM, so the chunk size doesn't change its bits.
TEST(SoftcapCE, WeightGradAccumulates) {
  torch::manual_seed(0);
  const int64_t N = 512, C = 128, vocab = 1000, padded = 1024;
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const auto w0 = torch::randn({padded, C}, opts) * 0.5;
  const std::array<torch::Tensor, 2> xs{
        torch::randn({N, C}, opts).to(torch::kBFloat16), torch::randn({N, C}, opts).to(torch::kBFloat16)};
  const auto targets = torch::randint(0, vocab, {N}, opts.dtype(torch::kInt64));
  const std::array<double, 2> scales{0.5, 0.25}; // e.g. 1 / grad_accum_steps
  for (const int mode : {0, 1, 2}) {             // bf16, tensorwise, MX
    const auto recipe = mode == 2 ? Fp8Recipe::Mx : Fp8Recipe::Tensorwise;
    const auto loss = [&](const torch::Tensor& w, int i, int64_t chunk) {
      return softcap_cross_entropy(xs[i], w, targets, vocab, 15, chunk, LossReduction::Mean, mode > 0, nullptr, recipe);
    };
    // each micro-step alone
    std::array<torch::Tensor, 2> single;
    for (int i = 0; i < 2; ++i) {
      auto w = w0.clone().requires_grad_();
      loss(w, i, N / 4).backward();
      single[i] = w.grad();
    }
    const auto want = single[0] * scales[0] + single[1] * scales[1];
    auto leaf = w0.clone().requires_grad_(), base = w0.clone().requires_grad_();
    for (int i = 0; i < 2; ++i) {
      (loss(leaf, i, N / 4) * scales[i]).backward();
      (loss(base * 1.0, i, N / 4) * scales[i]).backward(); // non-leaf
    }
    const auto what = "mode " + std::to_string(mode);
    EXPECT_LT(rel_norm_diff(leaf.grad(), want), 1e-6) << what;
    EXPECT_LT(rel_norm_diff(base.grad(), want), 1e-6) << what;
    if (mode == 2) {
      auto w = w0.clone().requires_grad_();
      loss(w, 0, N).backward();
      EXPECT_TRUE(torch::equal(w.grad(), single[0])) << "MX grad_w: 1 chunk vs 4";
    }
  }
}
