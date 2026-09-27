// FP8 quantization kernels vs fp8.py's torch ops (the model-level checks against Python are in test_gpt/test_train).
#include <gtest/gtest.h>

#include "nanochat/model/fp8.h"
#include "nanochat/model/gpt.h"

using namespace nanochat;

TEST(Fp8, FusedQuantizeMatchesTorchOps) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  // partial tiles in both dims, fp32 weights and bf16 activations
  for (const auto& [rows, cols] : {std::pair<int64_t, int64_t>{100, 24}, {1000, 136}, {64, 64}, {4097, 768}}) {
    for (const auto dtype : {torch::kBFloat16, torch::kFloat32}) {
      const auto x = (torch::randn({rows, cols}, opts) * 3).to(dtype);
      for (const auto format : {torch::kFloat8_e4m3fn, torch::kFloat8_e5m2}) {
        const auto got = quantize_fp8(x, format), want = quantize_fp8(x, format, false);
        const auto bits = [](const torch::Tensor& t) {
          return t.view(torch::kUInt8);
        };
        EXPECT_TRUE(torch::equal(bits(got.data), bits(want.data))) << rows << "x" << cols;
        EXPECT_TRUE(torch::equal(bits(got.data_t), bits(want.data.t().contiguous()))) << rows << "x" << cols;
        EXPECT_EQ(got.inv_scale.item<float>(), want.inv_scale.item<float>());
      }
    }
  }
}

TEST(Fp8, WeightCacheReusesUntilWeightChanges) {
  auto w = torch::randn({256, 128}, torch::TensorOptions().device(torch::kCUDA));
  Fp8WeightCache cache;
  cache.enabled = true;
  const auto a = quantize_fp8_weight(w, &cache), b = quantize_fp8_weight(w, &cache);
  EXPECT_TRUE(a.data.is_same(b.data));
  {
    torch::NoGradGuard no_grad;
    w.mul_(2); // as an optimizer step: bumps the version
  }
  const auto c = quantize_fp8_weight(w, &cache), want = quantize_fp8(w, torch::kFloat8_e4m3fn);
  EXPECT_FALSE(c.data.is_same(a.data));
  EXPECT_TRUE(torch::equal(c.data.view(torch::kUInt8), want.data.view(torch::kUInt8)));
  EXPECT_EQ(c.inv_scale.item<float>(), want.inv_scale.item<float>());
  cache.enabled = false;
  EXPECT_FALSE(quantize_fp8_weight(w, &cache).data.is_same(quantize_fp8_weight(w, &cache).data));
}

// Micro-steps, a weight update, another micro-step: the same bits with and without the cache.
TEST(Fp8, WeightCacheMatchesUncachedModel) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
  const GPTConfig
        config{.sequence_len = 256, .vocab_size = 1000, .n_layer = 2, .n_head = 2, .n_kv_head = 2, .n_embd = 256};
  torch::manual_seed(0);
  const auto ids = torch::randint(0, config.vocab_size, {3, 2, 257}, opts);
  auto run = [&](bool cache) {
    torch::manual_seed(0);
    GPT model(config);
    model->init_weights();
    model->set_fused(true);
    model->set_fp8(true);
    model->set_loss_chunk_rows(128);
    for (const auto& m : model->modules(false)) {
      if (auto* linear = dynamic_cast<LinearImpl*>(m.get()))
        linear->fp8_cache.enabled = cache;
      if (auto* attn = dynamic_cast<CausalSelfAttentionImpl*>(m.get()))
        attn->qkv_cache.enabled = cache;
    }
    std::vector<torch::Tensor> out;
    for (int64_t step = 0; step < 3; ++step) {
      const auto batch = ids[step];
      auto loss = model->forward(batch.slice(1, 0, -1).contiguous(), batch.slice(1, 1).contiguous());
      loss.backward();
      out.push_back(loss.detach());
      if (step == 1) {
        torch::NoGradGuard no_grad;
        for (auto& p : model->parameters())
          p.sub_(p.grad().sign() * 1e-2);
      }
    }
    for (const auto& p : model->parameters())
      out.push_back(p.grad().clone());
    return out;
  };
  const auto got = run(true), want = run(false);
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i)
    EXPECT_TRUE(torch::equal(got[i], want[i])) << i;
}
