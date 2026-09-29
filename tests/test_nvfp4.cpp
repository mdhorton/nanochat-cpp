// Simulated NVFP4 (nvfp4_sim.h): the fake-quantization kernel vs torch ops, its rounding modes and blocks, and the
// simulated Linear's outputs and gradients. Real NVFP4 (nvfp4.h): conversion, the kernels that write it directly, GEMM
// and weight gradients.
#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/gpt.h"
#include "nanochat/model/mx_gemm.h"
#include "nanochat/model/nvfp4.h"
#include "nanochat/model/nvfp4_sim.h"
#include "nanochat/model/nvfp4_sim_kernel.h"
#include "nanochat/model/residual_norm.h"

using namespace nanochat;

namespace {

const auto kCuda = torch::TensorOptions().device(torch::kCUDA);

// rows with magnitudes over several octaves, so the block scales differ
torch::Tensor spread(int64_t R, int64_t C) {
  return torch::randn({R, C}, kCuda) * torch::exp(torch::randn({R, 1}, kCuda) * 2);
}

// NVFP4 round to nearest in torch ops
torch::Tensor reference(const torch::Tensor& x) {
  const int64_t R = x.size(0), C = x.size(1);
  const auto xf = x.to(torch::kFloat32).view({R, C / 16, 16});
  const auto enc = 2688.f / xf.abs().max();
  const auto bmax = xf.abs().amax(-1, true);
  const auto s = (bmax / 6 * enc).clamp_max(448).to(torch::kFloat8_e4m3fn).to(torch::kFloat32);
  const auto a = (xf.abs() * (enc / s)).clamp_max(6);
  // round to nearest, ties to the even mantissa: each threshold below overrides the ones above
  auto q = torch::full_like(a, 6.f);
  q = torch::where(a <= 5, 4.f, q);
  q = torch::where(a < 3.5, 3.f, q);
  q = torch::where(a <= 2.5, 2.f, q);
  q = torch::where(a < 1.75, 1.5f, q);
  q = torch::where(a <= 1.25, 1.f, q);
  q = torch::where(a < .75, .5f, q);
  q = torch::where(a <= .25, 0.f, q);
  return (torch::sign(xf) * (q * (s / enc))).to(torch::kBFloat16).view({R, C});
}

double rel_err(const torch::Tensor& a, const torch::Tensor& b) {
  const auto bf = b.to(torch::kFloat32);
  return ((a.to(torch::kFloat32) - bf).norm() / bf.norm()).item<double>();
}

} // namespace

TEST(Nvfp4, FakeQuantMatchesReference) {
  torch::manual_seed(0);
  for (const auto dtype : {torch::kBFloat16, torch::kFloat32}) {
    const auto x = spread(256, 512).to(dtype);
    EXPECT_TRUE(torch::equal(nvfp4_fake_quant(x), reference(x))) << dtype;
  }
  EXPECT_EQ(nvfp4_fake_quant(torch::zeros({16, 32}, kCuda)).abs().max().item<float>(), 0.f);
}

TEST(Nvfp4, StochasticRoundingIsUnbiased) {
  torch::manual_seed(0);
  const auto x = torch::rand({256, 1024}, kCuda) * 2 - 1;
  auto sum = torch::zeros_like(x);
  const int n = 256;
  for (int i = 0; i < n; ++i) {
    const auto q = nvfp4_fake_quant(x, true);
    if (i == 0)
      EXPECT_FALSE(torch::equal(q, nvfp4_fake_quant(x, true))); // fresh random bits per call
    sum += q.to(torch::kFloat32);
  }
  const double sr = (sum / n - x).abs().mean().item<double>();
  const double rtn = (nvfp4_fake_quant(x).to(torch::kFloat32) - x).abs().mean().item<double>();
  EXPECT_LT(sr, 0.2 * rtn);
}

// 16x16 blocks: W and W^T quantize to the same values, so forward and dgrad see one weight
TEST(Nvfp4, TwoDBlocksMatchTranspose) {
  torch::manual_seed(0);
  const auto w = spread(128, 256);
  EXPECT_TRUE(torch::equal(nvfp4_fake_quant(w, false, true).t(), nvfp4_fake_quant(w.t().contiguous(), false, true)));
}

TEST(Nvfp4, HadamardPreservesProducts) {
  torch::manual_seed(0);
  const auto a = torch::randn({32, 256}, kCuda), b = torch::randn({48, 256}, kCuda);
  EXPECT_LT(rel_err(torch::mm(nvfp4_rht(a), nvfp4_rht(b).t()), torch::mm(a, b.t())), 1e-5);
  EXPECT_GT(rel_err(nvfp4_rht(a), a), 0.5); // it does mix
}

// Outputs and both gradients near the bf16 matmul's, with the error of 4-bit operands (also with rht on every GEMM);
// MXFP8 when all GEMMs are off
TEST(Nvfp4, SimMatmulNearBf16) {
  torch::manual_seed(0);
  const auto x0 = torch::randn({256, 512}, kCuda).to(torch::kBFloat16);
  const auto w0 = torch::randn({384, 512}, kCuda) * 0.05;
  const auto go = torch::randn({256, 384}, kCuda).to(torch::kBFloat16);
  auto run = [&](const Nvfp4Options* o) {
    auto x = x0.clone().requires_grad_(), w = w0.clone().requires_grad_();
    const auto y = o != nullptr ? nvfp4_sim_matmul(x, w, *o) : torch::mm(x, w.to(torch::kBFloat16).t());
    y.backward(go);
    return std::array{y.detach(), x.grad(), w.grad()};
  };
  const auto ref = run(nullptr);
  const Nvfp4Options fp4, mx{.fwd = false, .dgrad = false, .wgrad = false},
        rht{.rht_fwd = true, .rht_dgrad = true, .sr_dgrad = false, .seed = 7};
  const auto a = run(&fp4), b = run(&mx), c = run(&rht);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(a[i].scalar_type(), ref[i].scalar_type()) << i;
    EXPECT_LT(rel_err(a[i], ref[i]), 0.2) << i;
    EXPECT_GT(rel_err(a[i], ref[i]), 0.02) << i;
    EXPECT_LT(rel_err(c[i], ref[i]), 0.2) << i;
    EXPECT_GT(rel_err(c[i], ref[i]), 0.02) << i;
    EXPECT_LT(rel_err(b[i], ref[i]), 0.05) << i;
    EXPECT_LT(rel_err(b[i], ref[i]), rel_err(a[i], ref[i]) / 2) << i;
  }
}

// GPT: the blocks' Linears between the skipped ones switch; a training step runs
TEST(Nvfp4, GptSetNvfp4) {
  torch::manual_seed(0);
  GPT model(
        GPTConfig{
              .sequence_len = 256,
              .vocab_size = 1000,
              .n_layer = 4,
              .n_head = 2,
              .n_kv_head = 2,
              .n_embd = 256,
              .window_pattern = "SSSL"});
  model->init_weights();
  model->set_fused(true);
  model->set_fp8_recipe(Fp8Recipe::Mx);
  model->set_fp8(true);
  const Nvfp4Options o;
  EXPECT_EQ(model->set_nvfp4(&o, 1, 1), 2 * 6);
  auto h = model->transformer->h;
  EXPECT_EQ(h[0]->as<BlockImpl>()->attn->c_q->nvfp4, nullptr);
  EXPECT_EQ(h[1]->as<BlockImpl>()->mlp->c_proj->nvfp4, &o);
  EXPECT_EQ(model->lm_head->nvfp4, nullptr);
  const auto idx = torch::randint(0, 1000, {2, 256}, kCuda.dtype(torch::kInt64));
  auto loss = model->forward(idx, idx.roll(-1, 1));
  EXPECT_NEAR(loss.item<float>(), std::log(1000.0), 0.5);
  loss.backward();
  EXPECT_TRUE(h[1]->as<BlockImpl>()->attn->c_q->weight.grad().isfinite().all().item<bool>());
  EXPECT_EQ(model->set_nvfp4(nullptr), 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Real NVFP4 (nvfp4.h): from MX operands, the same rounding as the simulation.

namespace {

struct Mx {
  torch::Tensor data, scale, values; // values: dequantized bf16
};

Mx mx_of(const torch::Tensor& x) {
  const auto q = quantize_mx(x, true, false);
  return {q.data, q.inv_scale, mx_to_bf16(q.data, q.inv_scale)};
}

} // namespace

TEST(Nvfp4, MxToNvfp4MatchesFakeQuant) {
  torch::manual_seed(0);
  const auto m = mx_of(spread(256, 512).to(torch::kBFloat16));
  const int64_t R = 256, C = 512;
  // round to nearest
  const auto t = mx_to_nvfp4(m.data, m.scale, false, false);
  EXPECT_EQ(t.amax.item<float>(), m.values.abs().max().item<float>());
  EXPECT_TRUE(torch::equal(nvfp4_to_bf16(t), nvfp4_fake_quant(m.values)));
  // stochastic, same seed as the simulation's kernel
  const uint64_t seed = 12345;
  auto fake = torch::empty_like(m.values);
  const auto amax = m.values.abs().max().to(torch::kFloat32);
  kernels::nvfp4_fake_quant(
        m.values.data_ptr(), true, R, C, amax.data_ptr<float>(), nullptr, true, seed, fake.data_ptr(),
        at::cuda::getCurrentCUDAStream().stream());
  EXPECT_TRUE(torch::equal(nvfp4_to_bf16(mx_to_nvfp4(m.data, m.scale, false, true, seed)), fake));
  // Hadamard: summation order differs from nvfp4_rht's matmul, so a few values may round the other way
  at::globalContext().setFloat32MatmulPrecision("highest");
  const auto real = nvfp4_to_bf16(mx_to_nvfp4(m.data, m.scale, true, false));
  const auto sim = nvfp4_fake_quant(nvfp4_rht(m.values));
  EXPECT_LT((real != sim).to(torch::kFloat32).mean().item<double>(), 1e-3);
  EXPECT_LT(rel_err(real, sim), 1e-2);
}

// The GEMM reads the data and scales as mx_to_nvfp4 lays them out: it matches the dequantized product
TEST(Nvfp4, GemmMatchesDequantized) {
  torch::manual_seed(0);
  const auto ma = mx_of(spread(256, 512).to(torch::kBFloat16)), mb = mx_of(spread(384, 512).to(torch::kBFloat16));
  const auto qa = mx_to_nvfp4(ma.data, ma.scale, true, true, 1), qb = mx_to_nvfp4(mb.data, mb.scale, true, false);
  at::globalContext().setFloat32MatmulPrecision("highest");
  const auto ref =
        torch::mm(nvfp4_to_bf16(qa).to(torch::kFloat32), nvfp4_to_bf16(qb).to(torch::kFloat32).t()).contiguous();
  auto out = torch::zeros({256, 384}, kCuda);
  nvfp4_gemm_f32(qa, qb, out, false);
  EXPECT_LT(rel_err(out, ref), 1e-2); // the reference's operands are rounded to bf16
  // accumulate with alpha
  auto acc = torch::ones({256, 384}, kCuda);
  nvfp4_gemm_f32(qa, qb, acc, true, torch::full({}, 2.f, kCuda));
  EXPECT_LT(rel_err(acc, 1 + 2 * out), 1e-6);
  // near the unquantized product with the Hadamard (it cancels), rounding to nearest (a stochastic draw's error varies
  // a lot here: a few rows dominate)
  nvfp4_gemm_f32(mx_to_nvfp4(ma.data, ma.scale, true, false), qb, out, false);
  EXPECT_LT(rel_err(out, torch::mm(ma.values.to(torch::kFloat32), mb.values.to(torch::kFloat32).t())), 0.35);
}

// ---------------------------------------------------------------------------------------------------------------
// Kernels writing NVFP4 transposes (Nvfp4Target): the simulation's bits at the power-of-two tensor scale they pick.

namespace {

// nvfp4_fake_quant of x at t's tensor scale
torch::Tensor fake_at(const torch::Tensor& x, const Nvfp4Tensor& t, bool stochastic = false, uint64_t seed = 0) {
  auto out = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
  kernels::nvfp4_fake_quant(
        x.data_ptr(), x.scalar_type() == torch::kBFloat16, x.size(0), x.size(1), t.amax.data_ptr<float>(), nullptr,
        stochastic, seed, out.data_ptr(), at::cuda::getCurrentCUDAStream().stream());
  return out;
}

// x's elements in blocks of 16 whose scale is normal in e4m3 (>= 2^-6) at q's tensor scale. Below, the simulation
// rounds the codes against the coarser subnormal scale; producers can't (the tensor scale comes after them).
torch::Tensor normal_blocks(const torch::Tensor& x, const Nvfp4Tensor& q) {
  const auto bmax = x.to(torch::kFloat32).view({x.size(0), -1, 16}).abs().amax(-1, true);
  return (bmax / 6 * (2688.f / q.amax) >= std::ldexp(1.f, -6)).expand({-1, -1, 16}).reshape(x.sizes());
}

// transposes into t through quantize_mx's kernel
Nvfp4Tensor quantize_t(const torch::Tensor& x, const Nvfp4Target& t) {
  kernels::MxOut out{}, out_t{};
  out_t.fp4 = nvfp4_out(t);
  kernels::quantize_mx(
        x.data_ptr(), true, x.size(0), x.size(1), out, out_t, false, at::cuda::getCurrentCUDAStream().stream());
  return nvfp4_finish(t);
}

struct CutlassScope {
  CutlassScope() {
    set_mx_gemm_backend(MxGemmBackend::Cutlass);
  }

  ~CutlassScope() {
    set_mx_gemm_backend(MxGemmBackend::Cublas);
  }
};

} // namespace

TEST(Nvfp4, ProducerMatchesFakeQuant) {
  torch::manual_seed(0);
  const auto x = spread(512, 256).t().contiguous().to(torch::kBFloat16); // (256 tokens, 512); its transpose's rows vary
  const auto x_t = x.t().contiguous();
  for (const bool stochastic : {false, true}) {
    const auto q = quantize_t(x, empty_nvfp4(512, 256, kCuda, false, stochastic, 777));
    // the tensor scale: a power of two, the largest block scale within (224, 448]
    const float k = std::log2(2688.f / q.amax.item<float>());
    EXPECT_EQ(k, std::round(k));
    const float smax = q.scale.view(torch::kFloat8_e4m3fn).to(torch::kFloat32).max().item<float>();
    EXPECT_GT(smax, 224.f);
    EXPECT_LE(smax, 448.f);
    const auto normal = normal_blocks(x_t, q);
    EXPECT_GT(normal.to(torch::kFloat32).mean().item<double>(), 0.9);
    EXPECT_TRUE(
          torch::equal(nvfp4_to_bf16(q).masked_select(normal), fake_at(x_t, q, stochastic, 777).masked_select(normal)))
          << stochastic;
  }
  // Hadamard: summation order differs from nvfp4_rht's matmul, so a few values may round the other way
  at::globalContext().setFloat32MatmulPrecision("highest");
  const auto q = quantize_t(x, empty_nvfp4(512, 256, kCuda, true, false));
  const auto rht = nvfp4_rht(x_t);
  const auto normal = normal_blocks(rht, q);
  const auto real = nvfp4_to_bf16(q), sim = fake_at(rht, q);
  EXPECT_LT((real != sim).masked_select(normal).to(torch::kFloat32).mean().item<double>(), 1e-3);
  EXPECT_LT(rel_err(real, sim), 1e-2);
  // zeros
  const auto z = quantize_t(torch::zeros_like(x), empty_nvfp4(512, 256, kCuda, true, true, 1));
  EXPECT_EQ(nvfp4_to_bf16(z).abs().max().item<float>(), 0.f);
}

// residual_norm_mx with NVFP4 weight gradients: n's transpose, input rounding (to nearest)
TEST(Nvfp4, ResidualNormWritesNvfp4) {
  torch::manual_seed(0);
  const auto opts = kCuda.dtype(torch::kBFloat16);
  const auto x = torch::randn({256, 512}, opts), r = torch::randn({256, 512}, opts);
  const Nvfp4Wgrad o{.rht = false};
  set_nvfp4_wgrad(&o);
  const auto m = residual_norm_mx(x, r);
  set_nvfp4_wgrad(nullptr);
  const auto n = residual_norm(x, r).second;
  ASSERT_TRUE(m.n_mx.fp4_t());
  const Nvfp4Tensor q{m.n_mx.data_t, m.n_mx.inv_scale_t, m.n_mx.amax_t};
  EXPECT_TRUE(torch::equal(nvfp4_to_bf16(q), fake_at(n.t().contiguous(), q)));
  EXPECT_TRUE(torch::equal(m.n_mx.data.view(torch::kUInt8), quantize_mx(n, true, false).data.view(torch::kUInt8)));
}

// The relu^2 GEMM epilogues' NVFP4 transposes vs quantize_mx's kernels on the same values: bit for bit (Hadamard and
// stochastic rounding too)
TEST(Nvfp4, ReluSquareEpiloguesWriteNvfp4) {
  torch::manual_seed(0);
  const CutlassScope scope;
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t M = 256, N = 384, K = 512;
  const auto ma = mx_of(spread(M, K).to(torch::kBFloat16)), mb = mx_of(spread(N, K).to(torch::kBFloat16));
  const auto target = [&] {
    return empty_nvfp4(N, M, kCuda, true, true, 4242);
  };
  const auto u8 = [](const Nvfp4Tensor& t) {
    return torch::cat({t.data.flatten(), t.scale.flatten(), t.amax.view({1}).view(torch::kUInt8)});
  };
  // forward: relu(h)^2's transpose
  auto h = torch::empty({M, N}, kCuda.dtype(torch::kBFloat16));
  const auto q = empty_mx(M, N, kCuda, true, false);
  const auto t = target();
  ASSERT_TRUE(mx_gemm_relu_square(h, q.data, q.inv_scale, {}, {}, ma.data, ma.scale, mb.data, mb.scale, nvfp4_out(t)));
  kernels::MxOut none{}, out_t{};
  const auto want = target();
  out_t.fp4 = nvfp4_out(want);
  kernels::quantize_mx(h.data_ptr(), true, M, N, none, out_t, true, stream);
  EXPECT_TRUE(torch::equal(u8(nvfp4_finish(t)), u8(nvfp4_finish(want))));
  // backward: dh's transpose from ga = a . b^T and h
  const auto dh = empty_mx(M, N, kCuda, true, false);
  const auto t_bwd = target();
  ASSERT_TRUE(mx_gemm_relu_square_bwd(
        ma.data, ma.scale, mb.data, mb.scale, h, dh.data, dh.inv_scale, {}, {}, nvfp4_out(t_bwd)));
  const auto ga = mx_gemm(ma.data, ma.scale, mb.data, mb.scale, torch::kBFloat16);
  const auto want_bwd = target();
  out_t.fp4 = nvfp4_out(want_bwd);
  kernels::quantize_mx_relu_square_bwd(ga.data_ptr(), h.data_ptr(), M, N, none, out_t, stream);
  EXPECT_TRUE(torch::equal(u8(nvfp4_finish(t_bwd)), u8(nvfp4_finish(want_bwd))));
}

// set_nvfp4_wgrad: the blocks' weight gradients go NVFP4, lm_head's stays MXFP8 (same bits)
TEST(Nvfp4, WgradInGpt) {
  torch::manual_seed(0);
  GPT model(
        GPTConfig{
              .sequence_len = 256,
              .vocab_size = 1000,
              .n_layer = 2,
              .n_head = 2,
              .n_kv_head = 2,
              .n_embd = 256,
              .window_pattern = "L"});
  model->init_weights();
  { // init zeroes the output projections, which would zero the other weights' gradients
    torch::NoGradGuard no_grad;
    for (auto& p : model->transformer->h->parameters())
      if (p.dim() == 2)
        p.normal_(0, 0.05);
  }
  model->set_fused(true);
  model->set_fp8_recipe(Fp8Recipe::Mx);
  model->set_fp8(true);
  model->set_loss_chunk_rows(512);
  const auto idx = torch::randint(0, 1000, {2, 256}, kCuda.dtype(torch::kInt64));
  auto grads = [&](const Nvfp4Wgrad* o) {
    set_nvfp4_wgrad(o);
    model->zero_grad();
    model->forward(idx, idx.roll(-1, 1)).backward();
    set_nvfp4_wgrad(nullptr);
    std::map<std::string, torch::Tensor> g;
    for (const auto& p : model->named_parameters(true))
      g[p.key()] = p.value().grad().clone();
    return g;
  };
  const auto mx = grads(nullptr);
  const Nvfp4Wgrad o;
  const auto fp4 = grads(&o);
  EXPECT_TRUE(torch::equal(fp4.at("lm_head.weight"), mx.at("lm_head.weight")));
  for (const auto* name :
       {"transformer.h.0.attn.c_q.weight", "transformer.h.1.mlp.c_fc.weight", "transformer.h.1.mlp.c_proj.weight",
        "transformer.h.0.attn.c_proj.weight"}) {
    EXPECT_FALSE(torch::equal(fp4.at(name), mx.at(name))) << name;
    EXPECT_LT(rel_err(fp4.at(name), mx.at(name)), 0.3) << name;
  }
}
