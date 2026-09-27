// FP8 quantization kernels vs fp8.py's torch ops (the model-level checks against Python are in test_gpt/test_train).
#include <gtest/gtest.h>

#include "nanochat/model/fp8.h"
#include "nanochat/model/gpt.h"

using namespace nanochat;

namespace {

auto bits(const torch::Tensor& t) {
  return t.view(torch::kUInt8);
}

double rel_norm_diff(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).norm() / (b64.norm() + 1e-30)).item<double>();
}

// MXFP8 in torch ops: x (R, C) -> e4m3 data, biased e8m0 exponents (R, C / 32) of 2^ceil(log2(amax / 448))
std::pair<torch::Tensor, torch::Tensor> mx_reference(const torch::Tensor& x) {
  const int64_t R = x.size(0), C = x.size(1);
  const auto b = x.to(torch::kFloat32).view({R, C / 32, 32});
  const auto amax_bits = b.abs().amax(-1).view(torch::kInt32);
  const auto e = (torch::bitwise_right_shift(amax_bits, 23) - 8 + (amax_bits.bitwise_and(0x7fffff) > 0x600000))
                       .clamp(0, 253)
                       .to(torch::kInt32);
  const auto mult = torch::bitwise_left_shift(254 - e, 23).to(torch::kInt32).view(torch::kFloat32);
  return {(b * mult.unsqueeze(-1)).to(torch::kFloat8_e4m3fn).view({R, C}), e.to(torch::kUInt8)};
}

// (R, B) scales -> cuBLAS's swizzled layout: 128x4 tiles, (row % 32) * 16 + (row / 32 % 4) * 4 + block % 4
torch::Tensor swizzle(const torch::Tensor& s) {
  const int64_t R = s.size(0), B = s.size(1);
  return s.view({R / 128, 4, 32, B / 4, 4}).permute({0, 3, 2, 1, 4}).reshape(-1);
}

// dequantized MX data (R, C) with its swizzled scales
torch::Tensor mx_dequantize(const torch::Tensor& data, const torch::Tensor& scale) {
  const int64_t R = data.size(0), C = data.size(1);
  const auto e = bits(scale).view({R / 128, C / 128, 32, 4, 4}).permute({0, 3, 2, 1, 4}).reshape({R, C / 32});
  const auto mult = torch::pow(2.0, e.to(torch::kFloat32) - 127);
  return (data.to(torch::kFloat32).view({R, C / 32, 32}) * mult.unsqueeze(-1)).view({R, C});
}

} // namespace

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
  auto run = [&](bool cache, Fp8Recipe recipe) {
    torch::manual_seed(0);
    GPT model(config);
    model->init_weights();
    model->set_fused(true);
    model->set_fp8(true);
    model->set_fp8_recipe(recipe);
    model->set_attention(Attention::SDPA); // FA2's backward sums dq with atomics: not bit-reproducible
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
  for (const auto recipe : {Fp8Recipe::Tensorwise, Fp8Recipe::Mx}) {
    const auto got = run(true, recipe), want = run(false, recipe);
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i)
      EXPECT_TRUE(torch::equal(got[i], want[i])) << i << " mx=" << (recipe == Fp8Recipe::Mx);
  }
}

TEST(Fp8, MxQuantizeMatchesTorchOps) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  for (const auto& [rows, cols] : {std::pair<int64_t, int64_t>{128, 128}, {256, 384}, {384, 1024}}) {
    // magnitudes from 1e-6 to 1e6 across columns, a zero block, a block whose max has mantissa above 1.75
    auto x = torch::randn({rows, cols}, opts) * torch::logspace(-6, 6, cols, 10.0, opts);
    x.slice(0, 0, 1).slice(1, 0, 32).zero_();
    x[1][40] = 448 * 1.8;
    for (const auto dtype : {torch::kBFloat16, torch::kFloat32}) {
      const auto xd = x.to(dtype);
      for (const bool relu_square : {false, true}) {
        if (relu_square && dtype != torch::kBFloat16)
          continue;
        const auto src = relu_square ? torch::relu(xd).square() : xd;
        const auto got = quantize_mx(xd, true, true, relu_square);
        const auto [data, e] = mx_reference(src);
        const auto [data_t, e_t] = mx_reference(src.t().contiguous());
        const auto what = std::to_string(rows) + "x" + std::to_string(cols) + (relu_square ? " relu^2" : "");
        EXPECT_TRUE(torch::equal(bits(got.data), bits(data))) << what;
        EXPECT_TRUE(torch::equal(bits(got.inv_scale), swizzle(e))) << what;
        EXPECT_TRUE(torch::equal(bits(got.data_t), bits(data_t))) << what;
        EXPECT_TRUE(torch::equal(bits(got.inv_scale_t), swizzle(e_t))) << what;
      }
    }
  }
}

// The three MX GEMMs match the dequantized operands, and land closer to fp32 than tensorwise.
TEST(Fp8, MxMatmulMatchesDequantized) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  const int64_t N = 512, in = 256, out = 384;
  const auto x0 = (torch::randn({N, in}, opts) * torch::logspace(-2, 2, in, 10.0, opts)).to(torch::kBFloat16);
  const auto w0 = torch::randn({out, in}, opts) * 0.05;
  const auto g0 = (torch::randn({N, out}, opts) * torch::logspace(-3, 1, out, 10.0, opts)).to(torch::kBFloat16);
  auto run = [&](std::optional<Fp8Recipe> recipe) {
    auto x = (recipe ? x0 : x0.to(torch::kFloat32)).clone().requires_grad_();
    auto w = w0.clone().requires_grad_();
    auto y = recipe ? fp8_matmul(x, w, nullptr, *recipe) : torch::mm(x, w.t());
    y.backward(recipe ? g0 : g0.to(torch::kFloat32));
    return std::tuple{y.detach(), x.grad(), w.grad()};
  };
  const auto [y, dx, dw] = run(Fp8Recipe::Mx);
  const auto [y_tw, dx_tw, dw_tw] = run(Fp8Recipe::Tensorwise);
  const auto [y_ref, dx_ref, dw_ref] = run(std::nullopt);

  const auto xq = quantize_mx(x0), wq = quantize_mx(w0), gq = quantize_mx(g0);
  const auto deq = [](const torch::Tensor& d, const torch::Tensor& s) {
    return mx_dequantize(d, s);
  };
  EXPECT_LT(rel_norm_diff(y, torch::mm(deq(xq.data, xq.inv_scale), deq(wq.data, wq.inv_scale).t())), 4e-3);
  EXPECT_LT(rel_norm_diff(dx, torch::mm(deq(gq.data, gq.inv_scale), deq(wq.data_t, wq.inv_scale_t).t())), 4e-3);
  EXPECT_LT(rel_norm_diff(dw, torch::mm(deq(gq.data_t, gq.inv_scale_t), deq(xq.data_t, xq.inv_scale_t).t())), 4e-3);
  for (const auto& [got, tw, ref, name] :
       {std::tuple{y, y_tw, y_ref, "y"}, std::tuple{dx, dx_tw, dx_ref, "dx"}, std::tuple{dw, dw_tw, dw_ref, "dw"}})
    EXPECT_LT(rel_norm_diff(got, ref), rel_norm_diff(tw, ref)) << name;
}

// Fused model (merged q/k/v, relu^2 MLP, chunked loss) with MX: close to bf16, closer than tensorwise. Random
// matrices (zero-init c_proj would zero most gradients); the scalars' gradients cancel too much to compare.
TEST(Fp8, MxModelCloseToBf16) {
  const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
  const GPTConfig
        config{.sequence_len = 256, .vocab_size = 1000, .n_layer = 2, .n_head = 2, .n_kv_head = 2, .n_embd = 256};
  torch::manual_seed(0);
  const auto ids = torch::randint(0, config.vocab_size, {2, 257}, opts);
  auto run = [&](std::optional<Fp8Recipe> recipe) {
    torch::manual_seed(0);
    GPT model(config);
    model->init_weights();
    {
      torch::NoGradGuard no_grad;
      for (auto& p : model->parameters())
        if (p.dim() == 2 && p.size(0) >= 128)
          p.normal_(0, 0.05);
    }
    model->set_fused(true);
    model->set_loss_chunk_rows(128);
    if (recipe) {
      model->set_fp8(true);
      model->set_fp8_recipe(*recipe);
    }
    auto loss = model->forward(ids.slice(1, 0, -1).contiguous(), ids.slice(1, 1).contiguous());
    loss.backward();
    std::vector<torch::Tensor> out{loss.detach()};
    for (const auto& p : model->parameters())
      if (p.dim() == 2 && p.size(0) >= 128)
        out.push_back(p.grad().clone());
    return out;
  };
  const auto mx = run(Fp8Recipe::Mx), tw = run(Fp8Recipe::Tensorwise), ref = run(std::nullopt);
  double mx_err = 0, tw_err = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    mx_err = std::max(mx_err, rel_norm_diff(mx[i], ref[i]));
    tw_err = std::max(tw_err, rel_norm_diff(tw[i], ref[i]));
  }
  EXPECT_LT(mx_err, 0.2); // FP8 through random matrices: about 15%; a layout bug gives > 100%
  EXPECT_LT(mx_err, tw_err);
  std::cout << "max rel err vs bf16: mx " << mx_err << ", tensorwise " << tw_err << "\n";
}
