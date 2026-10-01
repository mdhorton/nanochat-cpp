// Simulated NVFP4 (nvfp4_sim.h): the fake-quantization kernel vs torch ops, its rounding modes and blocks, and the
// simulated Linear's outputs and gradients. Real NVFP4 (nvfp4.h): conversion, the kernels that write it directly, GEMM
// and weight gradients.
#include <optional>

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

TEST(Nvfp4, EdenRotationIsOrthonormal) {
  const auto h = nvfp4_eden_rotation(torch::kCUDA, 1);
  EXPECT_LT((torch::mm(h, h.t()) - torch::eye(128, kCuda)).abs().max().item<float>(), 1e-5);
  EXPECT_FALSE(torch::equal(h, nvfp4_eden_rotation(torch::kCUDA, 1))); // fresh signs per call
  // other groups, and fixed signs: the same seed gives the same rotation
  const auto h64 = nvfp4_eden_rotation(torch::kCUDA, 5, 64, false);
  EXPECT_LT((torch::mm(h64, h64.t()) - torch::eye(64, kCuda)).abs().max().item<float>(), 1e-5);
  EXPECT_TRUE(torch::equal(h64, nvfp4_eden_rotation(torch::kCUDA, 5, 64, false)));
  EXPECT_FALSE(torch::equal(h64, nvfp4_eden_rotation(torch::kCUDA, 6, 64, false)));
}

// With a 64-group rotation, each 64-group's <x.h, dequantized> is ||x.h||² in expectation
TEST(Nvfp4, EdenGroup64IsUnbiased) {
  torch::manual_seed(0);
  const int64_t R = 64, C = 1024, G = 64;
  const auto x = torch::randn({R, C}, kCuda);
  const auto h = nvfp4_eden_rotation(torch::kCUDA, 0, G);
  const auto xh = torch::matmul(x.view({R, C / G, G}), h);
  const auto norm2 = (xh * xh).sum(-1);
  auto mean = torch::zeros_like(norm2);
  const int n = 64;
  for (int i = 0; i < n; ++i) {
    const auto q = nvfp4_eden(x, h);
    mean += (xh * (q.values.to(torch::kFloat32) * q.scale).view({R, C / G, G})).sum(-1) / norm2 / n;
  }
  EXPECT_LT((mean - 1).abs().max().item<float>(), 0.04);
  EXPECT_LT((mean - 1).mean().abs().item<float>(), 0.003);
}

// Each 128-group's <x.h, dequantized> is ||x.h||² in expectation (within one e4m3 scale step per call)
TEST(Nvfp4, EdenCorrectionIsUnbiased) {
  torch::manual_seed(0);
  const int64_t R = 64, C = 1024;
  const auto x = torch::randn({R, C}, kCuda);
  const auto h = nvfp4_eden_rotation(torch::kCUDA);
  const auto xh = torch::matmul(x.view({R, C / 128, 128}), h);
  const auto norm2 = (xh * xh).sum(-1);
  auto mean = torch::zeros_like(norm2);
  const int n = 64;
  for (int i = 0; i < n; ++i) {
    const auto q = nvfp4_eden(x, h);
    const auto ratio = (xh * (q.values.to(torch::kFloat32) * q.scale).view({R, C / 128, 128})).sum(-1) / norm2;
    EXPECT_LT((ratio - 1).abs().max().item<float>(), 0.13) << i;
    mean += ratio / n;
  }
  EXPECT_LT((mean - 1).abs().max().item<float>(), 0.03);
  EXPECT_LT((mean - 1).mean().abs().item<float>(), 0.002);
}

// A product's estimates, each with a fresh rotation and rounding, average towards the exact product
TEST(Nvfp4, EdenProductAveragesToExact) {
  torch::manual_seed(0);
  const auto a = torch::randn({64, 1024}, kCuda).to(torch::kBFloat16), b = torch::randn({48, 1024}, kCuda);
  const auto exact = torch::mm(a.to(torch::kFloat32), b.t());
  auto sum = torch::zeros_like(exact);
  double single = 0;
  const int n = 64;
  for (int i = 0; i < n; ++i) {
    const auto h = nvfp4_eden_rotation(torch::kCUDA);
    const auto qa = nvfp4_eden(a, h), qb = nvfp4_eden(b, h);
    const auto est = at::mm(qa.values, qb.values.t(), torch::kFloat32) * (qa.scale * qb.scale);
    single += rel_err(est, exact) / n;
    sum += est;
  }
  EXPECT_LT(single, 0.2);
  EXPECT_LT(rel_err(sum / n, exact), 0.25 * single);
}

// Outputs and both gradients near the bf16 matmul's, with the error of 4-bit operands (also with rht on every GEMM, and
// with MS-EDEN on the backward GEMMs); MXFP8 when all GEMMs are off
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
        rht{.rht_fwd = true, .rht_dgrad = true, .sr_dgrad = false, .seed = 7},
        eden{.eden_dgrad = true, .eden_wgrad = true, .seed = 3},
        eden64{.eden_dgrad = true, .eden_wgrad = true, .eden_group = 64, .eden_fixed_signs = true, .seed = 3};
  const auto a = run(&fp4), b = run(&mx), c = run(&rht), e = run(&eden), e64 = run(&eden64);
  EXPECT_TRUE(torch::equal(e[0], a[0])); // forward unchanged
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(e[i].scalar_type(), ref[i].scalar_type()) << i;
    if (i > 0) {
      EXPECT_LT(rel_err(e[i], ref[i]), 0.2) << i;
      EXPECT_GT(rel_err(e[i], ref[i]), 0.02) << i;
    }
    EXPECT_EQ(a[i].scalar_type(), ref[i].scalar_type()) << i;
    EXPECT_LT(rel_err(a[i], ref[i]), 0.2) << i;
    EXPECT_GT(rel_err(a[i], ref[i]), 0.02) << i;
    EXPECT_LT(rel_err(c[i], ref[i]), 0.2) << i;
    EXPECT_GT(rel_err(c[i], ref[i]), 0.02) << i;
    EXPECT_LT(rel_err(b[i], ref[i]), 0.05) << i;
    EXPECT_LT(rel_err(b[i], ref[i]), rel_err(a[i], ref[i]) / 2) << i;
    if (i > 0) {
      EXPECT_LT(rel_err(e64[i], ref[i]), 0.2) << i;
      EXPECT_GT(rel_err(e64[i], ref[i]), 0.02) << i;
    }
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
  // eden_skip: by the module name's last parts; ids distinct
  EXPECT_EQ(model->set_nvfp4(&o, 1, 1, {"c_fc", "mlp.c_proj"}), 2 * 6);
  EXPECT_FALSE(h[1]->as<BlockImpl>()->mlp->c_fc->nvfp4_eden);
  EXPECT_FALSE(h[1]->as<BlockImpl>()->mlp->c_proj->nvfp4_eden);
  EXPECT_TRUE(h[1]->as<BlockImpl>()->attn->c_proj->nvfp4_eden);
  EXPECT_NE(h[1]->as<BlockImpl>()->attn->c_q->nvfp4_id, h[2]->as<BlockImpl>()->attn->c_q->nvfp4_id);
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

// cuBLASLt reads the operands as CUTLASS lays them out (VEC16_UE4M3 scales, the tensor scales as a device alpha):
// the same product up to summation order
TEST(Nvfp4, GemmCublasMatchesCutlass) {
  torch::manual_seed(0);
  const auto ma = mx_of(spread(256, 512).to(torch::kBFloat16)), mb = mx_of(spread(384, 512).to(torch::kBFloat16));
  const auto qa = mx_to_nvfp4(ma.data, ma.scale, true, true, 1), qb = mx_to_nvfp4(mb.data, mb.scale, true, false);
  auto cutlass = torch::zeros({256, 384}, kCuda), cublas = torch::zeros({256, 384}, kCuda);
  set_nvfp4_gemm_backend(Nvfp4GemmBackend::Cutlass);
  nvfp4_gemm_f32(qa, qb, cutlass, false);
  const auto cutlass_bf16 = nvfp4_gemm(qa, qb);
  set_nvfp4_gemm_backend(Nvfp4GemmBackend::Cublas);
  nvfp4_gemm_f32(qa, qb, cublas, false);
  const auto cublas_bf16 = nvfp4_gemm(qa, qb);
  set_nvfp4_gemm_backend(Nvfp4GemmBackend::Auto); // restored below
  EXPECT_GT(cutlass.abs().max().item<float>(), 0.f);
  EXPECT_LT(rel_err(cublas, cutlass), 1e-6);
  EXPECT_LT(rel_err(cublas_bf16.to(torch::kFloat32), cutlass_bf16.to(torch::kFloat32)), 1e-2);
  // accumulate with alpha
  auto acc = torch::ones({256, 384}, kCuda);
  nvfp4_gemm_cublas(qa, qb, acc, true, torch::full({}, 2.f, kCuda));
  EXPECT_LT(rel_err(acc, 1 + 2 * cutlass), 1e-6);
  // Auto picks per shape and runs
  auto out = torch::zeros({256, 384}, kCuda);
  nvfp4_gemm_f32(qa, qb, out, false);
  EXPECT_LT(rel_err(out, cutlass), 1e-6);
  set_nvfp4_gemm_backend(Nvfp4GemmBackend::Cutlass);
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
    EXPECT_GT(normal.to(torch::kFloat32).mean().item<double>(), 0.7);
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
  const Nvfp4Backward o{.rht = false};
  set_nvfp4_backward(&o);
  const auto m = residual_norm_mx(x, r);
  set_nvfp4_backward(nullptr);
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

// ---------------------------------------------------------------------------------------------------------------
// NVFP4 dgrad: the gradient's rows and the weights' transposes, blocks along the output features.

// quantize_mx's rows as NVFP4: the simulation's bits (round to nearest and stochastic)
TEST(Nvfp4, ProducerRowsMatchFakeQuant) {
  torch::manual_seed(0);
  const auto x = spread(256, 512).to(torch::kBFloat16);
  for (const bool stochastic : {false, true}) {
    const auto t = empty_nvfp4(256, 512, kCuda, false, stochastic, 99);
    kernels::MxOut out{}, none{};
    out.fp4 = nvfp4_out(t);
    kernels::quantize_mx(x.data_ptr(), true, 256, 512, out, none, false, at::cuda::getCurrentCUDAStream().stream());
    const auto q = nvfp4_finish(t);
    const auto normal = normal_blocks(x, q);
    EXPECT_GT(normal.to(torch::kFloat32).mean().item<double>(), 0.7);
    EXPECT_TRUE(
          torch::equal(nvfp4_to_bf16(q).masked_select(normal), fake_at(x, q, stochastic, 99).masked_select(normal)))
          << stochastic;
  }
}

// bf16 output: the fp32 GEMM's, rounded
TEST(Nvfp4, GemmBf16MatchesF32) {
  torch::manual_seed(0);
  const auto ma = mx_of(spread(256, 512).to(torch::kBFloat16)), mb = mx_of(spread(384, 512).to(torch::kBFloat16));
  const auto qa = mx_to_nvfp4(ma.data, ma.scale, false, true, 3), qb = mx_to_nvfp4(mb.data, mb.scale, false, false);
  auto out = torch::empty({256, 384}, kCuda);
  nvfp4_gemm_f32(qa, qb, out, false);
  EXPECT_TRUE(torch::equal(nvfp4_gemm(qa, qb), out.to(torch::kBFloat16)));
}

// The NVFP4 relu^2 dgrad GEMM's epilogue (dh rows and transpose) vs nvfp4_gemm then quantize_mx's kernel: bit for
// bit, with the rows stochastic (NVFP4 dgrad) or MS-EDEN
TEST(Nvfp4, ReluSquareBwdEpilogueNvfp4) {
  torch::manual_seed(0);
  const CutlassScope scope;
  const int64_t M = 256, N = 384, K = 512;
  const auto ma = mx_of(spread(M, K).to(torch::kBFloat16)), mb = mx_of(spread(N, K).to(torch::kBFloat16));
  const auto a = mx_to_nvfp4(ma.data, ma.scale, false, true, 5), b = mx_to_nvfp4(mb.data, mb.scale, false, false);
  const auto h = torch::randn({M, N}, kCuda.dtype(torch::kBFloat16));
  const auto u8 = [](const Nvfp4Tensor& t) {
    return torch::cat({t.data.flatten(), t.scale.flatten(), t.amax.view({1}).view(torch::kUInt8)});
  };
  for (const bool eden : {false, true}) {
    const auto run = [&](bool fused) {
      const auto rows = eden ? empty_nvfp4(M, N, kCuda, false, false, 11, true, 0x9e3779b97f4a7c15ull)
                             : empty_nvfp4(M, N, kCuda, false, true, 11);
      const auto cols = empty_nvfp4(N, M, kCuda, true, true, 12);
      kernels::MxOut out{}, out_t{};
      out.fp4 = nvfp4_out(rows), out_t.fp4 = nvfp4_out(cols);
      if (fused)
        EXPECT_TRUE(nvfp4_gemm_relu_square_bwd(a, b, h, out, out_t));
      else
        kernels::quantize_mx_relu_square_bwd(
              nvfp4_gemm(a, b).data_ptr(), h.data_ptr(), M, N, out, out_t, at::cuda::getCurrentCUDAStream().stream());
      return std::pair{u8(nvfp4_finish(rows)), u8(nvfp4_finish(cols))};
    };
    const auto f = run(true), u = run(false);
    EXPECT_TRUE(torch::equal(f.first, u.first)) << "eden " << eden;
    EXPECT_TRUE(torch::equal(f.second, u.second)) << "eden " << eden;
  }
}

// set_nvfp4_backward: the blocks' weight gradients go NVFP4, lm_head's stays MXFP8 (same bits)
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
  auto grads = [&](const Nvfp4Backward* o) {
    set_nvfp4_backward(o);
    model->zero_grad();
    model->forward(idx, idx.roll(-1, 1)).backward();
    set_nvfp4_backward(nullptr);
    std::map<std::string, torch::Tensor> g;
    for (const auto& p : model->named_parameters(true))
      g[p.key()] = p.value().grad().clone();
    return g;
  };
  const auto mx = grads(nullptr);
  const Nvfp4Backward o;
  const auto fp4 = grads(&o);
  EXPECT_TRUE(torch::equal(fp4.at("lm_head.weight"), mx.at("lm_head.weight")));
  for (const auto* name :
       {"transformer.h.0.attn.c_q.weight", "transformer.h.1.mlp.c_fc.weight", "transformer.h.1.mlp.c_proj.weight",
        "transformer.h.0.attn.c_proj.weight"}) {
    EXPECT_FALSE(torch::equal(fp4.at(name), mx.at(name))) << name;
    EXPECT_LT(rel_err(fp4.at(name), mx.at(name)), 0.3) << name;
  }
  // NVFP4 dgrad (with and without wgrad): the last block's c_proj weights (upstream of every dgrad) keep their bits
  for (const bool wgrad : {false, true}) {
    const Nvfp4Backward d{.wgrad = wgrad, .dgrad = true};
    const auto g = grads(&d);
    EXPECT_TRUE(torch::equal(g.at("lm_head.weight"), mx.at("lm_head.weight")));
    EXPECT_EQ(
          torch::equal(g.at("transformer.h.1.mlp.c_proj.weight"), mx.at("transformer.h.1.mlp.c_proj.weight")), !wgrad);
    for (const auto* name :
         {"transformer.h.0.attn.c_q.weight", "transformer.h.1.mlp.c_fc.weight", "transformer.h.0.mlp.c_proj.weight",
          "transformer.h.0.attn.c_proj.weight", "transformer.wte.weight"}) {
      EXPECT_FALSE(torch::equal(g.at(name), mx.at(name))) << name << " wgrad " << wgrad;
      EXPECT_LT(rel_err(g.at(name), mx.at(name)), 0.4) << name << " wgrad " << wgrad;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// NVFP4 forward: the inputs' rows, weights in 16x16 blocks.

// 16x16 blocks: the simulation's bits (weight_2d) at the tensor scale picked; the transpose holds the same values
TEST(Nvfp4, TwoDWeightMatchesFakeQuant) {
  torch::manual_seed(0);
  const int64_t R = 256, C = 384;
  for (const auto dtype : {torch::kFloat32, torch::kBFloat16}) {
    const auto w = (spread(C, R).t().contiguous() * 0.05).to(dtype); // magnitudes vary along both dims
    const auto rows = empty_nvfp4(R, C, kCuda, false, false), cols = empty_nvfp4(C, R, kCuda, false, false);
    quantize_nvfp4_2d(w, nvfp4_out(rows), nvfp4_out(cols));
    const auto [q, q_t] = nvfp4_finish(rows, cols);
    EXPECT_TRUE(torch::equal(nvfp4_to_bf16(q_t), nvfp4_to_bf16(q).t())) << dtype;
    const auto bmax = w.to(torch::kFloat32).abs().view({R / 16, 16, C / 16, 16}).amax({1, 3}).repeat_interleave(16, 0);
    auto fake = torch::empty({R, C}, kCuda.dtype(torch::kBFloat16));
    kernels::nvfp4_fake_quant(
          w.data_ptr(), dtype == torch::kBFloat16, R, C, q.amax.data_ptr<float>(), bmax.data_ptr<float>(), false, 0,
          fake.data_ptr(), at::cuda::getCurrentCUDAStream().stream());
    const auto normal = (bmax / 6 * (2688.f / q.amax) >= std::ldexp(1.f, -6)).repeat_interleave(16, 1);
    EXPECT_GT(normal.to(torch::kFloat32).mean().item<double>(), 0.9) << dtype;
    EXPECT_TRUE(torch::equal(nvfp4_to_bf16(q).masked_select(normal), fake.masked_select(normal))) << dtype;
  }
}

// residual_norm_mx's NVFP4 rows: quantize_mx's of n, bit for bit
TEST(Nvfp4, ResidualNormWritesNvfp4Rows) {
  torch::manual_seed(0);
  const auto opts = kCuda.dtype(torch::kBFloat16);
  const auto x = torch::randn({256, 512}, opts), r = torch::randn({256, 512}, opts);
  const auto m = residual_norm_mx(x, r, {}, {}, {}, 0, {}, 0, true);
  const auto n = residual_norm(x, r).second;
  ASSERT_TRUE(m.n_mx.fp4());
  const auto want = quantize_mx(n, true, true, false, Nvfp4Role::FwdInput);
  EXPECT_TRUE(torch::equal(m.n_mx.data, want.data));
  EXPECT_TRUE(torch::equal(m.n_mx.inv_scale, want.inv_scale));
  EXPECT_TRUE(torch::equal(m.n_mx.amax, want.amax));
  EXPECT_TRUE(torch::equal(m.n_mx.data_t.view(torch::kUInt8), want.data_t.view(torch::kUInt8)));
}

// The NVFP4 relu^2 forward GEMM's epilogue (h, relu^2 rows and transpose) vs nvfp4_gemm then quantize_mx's kernel: bit
// for bit
TEST(Nvfp4, ReluSquareFwdEpilogueNvfp4) {
  torch::manual_seed(0);
  const CutlassScope scope;
  const int64_t M = 256, N = 384, K = 512;
  const auto ma = mx_of(spread(M, K).to(torch::kBFloat16)), mb = mx_of(spread(N, K).to(torch::kBFloat16));
  const auto a = mx_to_nvfp4(ma.data, ma.scale, false, false), b = mx_to_nvfp4(mb.data, mb.scale, false, false);
  const auto u8 = [](const Nvfp4Tensor& t) {
    return torch::cat({t.data.flatten(), t.scale.flatten(), t.amax.view({1}).view(torch::kUInt8)});
  };
  const auto run = [&](bool fused, bool fp4_t) {
    auto h = torch::empty({M, N}, kCuda.dtype(torch::kBFloat16));
    const auto rows = empty_nvfp4(M, N, kCuda, false, false);
    const auto cols = fp4_t ? empty_nvfp4(N, M, kCuda, true, true, 13) : Nvfp4Target{};
    const auto mx_t = empty_mx(M, N, kCuda, false, true);
    kernels::MxOut out{}, out_t{};
    out.fp4 = nvfp4_out(rows);
    if (fp4_t)
      out_t.fp4 = nvfp4_out(cols);
    else
      out_t = mx_out(mx_t.data_t, mx_t.inv_scale_t, 0, 0);
    if (fused)
      EXPECT_TRUE(nvfp4_gemm_relu_square(a, b, h, out, out_t));
    else {
      h = nvfp4_gemm(a, b);
      quantize_mx_into(h, true, out, out_t);
    }
    return std::tuple{
          h, u8(nvfp4_finish(rows)),
          fp4_t ? u8(nvfp4_finish(cols))
                : torch::cat({mx_t.data_t.view(torch::kUInt8).flatten(), mx_t.inv_scale_t.view(torch::kUInt8)})};
  };
  for (const bool fp4_t : {false, true}) {
    const auto f = run(true, fp4_t), u = run(false, fp4_t);
    EXPECT_TRUE(torch::equal(std::get<0>(f), std::get<0>(u))) << fp4_t;
    EXPECT_TRUE(torch::equal(std::get<1>(f), std::get<1>(u))) << fp4_t;
    EXPECT_TRUE(torch::equal(std::get<2>(f), std::get<2>(u))) << fp4_t;
  }
}

// fp8_matmul with fp4: the product of the dequantized NVFP4 operands; with NVFP4 dgrad the weight's transpose holds
// the forward's values
TEST(Nvfp4, FwdMatmulMatchesDequantized) {
  torch::manual_seed(0);
  const auto x = spread(256, 512).to(torch::kBFloat16);
  const auto w = torch::randn({384, 512}, kCuda) * 0.05;
  const auto y = fp8_matmul(x, w, nullptr, Fp8Recipe::Mx, true);
  const auto qx = quantize_mx(x, true, false, false, Nvfp4Role::FwdInput);
  const auto qw = quantize_fp8_weight(w, nullptr, Fp8Recipe::Mx, false, true);
  ASSERT_TRUE(qx.fp4() && qw.fp4() && !qw.fp4_t());
  at::globalContext().setFloat32MatmulPrecision("highest");
  const auto ref = torch::mm(
        nvfp4_to_bf16(qx.nvfp4()).to(torch::kFloat32), nvfp4_to_bf16(qw.nvfp4()).to(torch::kFloat32).t());
  EXPECT_LT(rel_err(y, ref), 1e-2);
  EXPECT_LT(rel_err(y, torch::mm(x.to(torch::kFloat32), w.t())), 0.2);
  const Nvfp4Backward d{.wgrad = false, .dgrad = true};
  set_nvfp4_backward(&d);
  const auto q2 = quantize_fp8_weight(w, nullptr, Fp8Recipe::Mx, true, true);
  set_nvfp4_backward(nullptr);
  ASSERT_TRUE(q2.fp4() && q2.fp4_t());
  EXPECT_TRUE(torch::equal(nvfp4_to_bf16(q2.nvfp4_t()), nvfp4_to_bf16(q2.nvfp4()).t()));
}

// set_nvfp4_fwd: the blocks' Linears between the skipped ones, not lm_head; a step runs, near MXFP8, with NVFP4
// backward GEMMs too
TEST(Nvfp4, FwdInGpt) {
  torch::manual_seed(0);
  const CutlassScope scope;
  GPT model(
        GPTConfig{
              .sequence_len = 256,
              .vocab_size = 1000,
              .n_layer = 3,
              .n_head = 2,
              .n_kv_head = 2,
              .n_embd = 256,
              .window_pattern = "L"});
  model->init_weights();
  {
    torch::NoGradGuard no_grad;
    for (auto& p : model->transformer->h->parameters())
      if (p.dim() == 2)
        p.normal_(0, 0.05);
  }
  model->set_fused(true);
  model->set_fp8_recipe(Fp8Recipe::Mx);
  model->set_fp8(true);
  model->set_attention(Attention::MX);
  model->set_loss_chunk_rows(512);
  const auto idx = torch::randint(0, 1000, {2, 256}, kCuda.dtype(torch::kInt64));
  const auto step = [&](const Nvfp4Backward* o) {
    set_nvfp4_backward(o);
    model->zero_grad();
    auto loss = model->forward(idx, idx.roll(-1, 1));
    loss.backward();
    set_nvfp4_backward(nullptr);
    return std::pair{loss.item<float>(), model->transformer->h[0]->as<BlockImpl>()->attn->c_q->weight.grad().clone()};
  };
  const auto mx = step(nullptr);
  EXPECT_EQ(model->set_nvfp4_fwd(true, 0, 1), 2 * 6);
  auto h = model->transformer->h;
  EXPECT_TRUE(h[1]->as<BlockImpl>()->mlp->c_proj->nvfp4_fwd);
  EXPECT_FALSE(h[2]->as<BlockImpl>()->attn->c_q->nvfp4_fwd);
  EXPECT_FALSE(model->lm_head->nvfp4_fwd);
  const Nvfp4Backward all{.dgrad = true};
  for (const auto* o : {static_cast<const Nvfp4Backward*>(nullptr), &all}) {
    const auto fp4 = step(o);
    EXPECT_NE(fp4.first, mx.first);
    EXPECT_NEAR(fp4.first, mx.first, 0.05);
    EXPECT_TRUE(fp4.second.isfinite().all().item<bool>());
    EXPECT_LT(rel_err(fp4.second, mx.second), 0.7); // the simulation's forward alone: ~0.4
  }
  EXPECT_EQ(model->set_nvfp4_fwd(false), 0);
  EXPECT_EQ(step(nullptr).first, mx.first);
}

// ---------------------------------------------------------------------------------------------------------------
// NVFP4 dgrad with MS-EDEN: the producers write both operands rotated along K in 64-groups by the GEMM's signs.

namespace {

// diag(signs) . H64 (entries +-1), fp32
torch::Tensor eden_rotation(uint64_t signs) {
  auto h = torch::ones({1, 1});
  for (int n = 1; n < 64; n *= 2)
    h = torch::cat({torch::cat({h, h}, 1), torch::cat({h, -h}, 1)}, 0);
  std::vector<float> s(64);
  for (int i = 0; i < 64; ++i)
    s[i] = (signs >> i) & 1u ? -1.f : 1.f;
  return (torch::tensor(s).view({64, 1}) * h).to(torch::kCUDA);
}

// rows a little apart in magnitude (block scales stay normal in e4m3)
torch::Tensor mild(int64_t R, int64_t C) {
  return torch::randn({R, C}, kCuda) * torch::exp(torch::randn({R, 1}, kCuda) * 0.5);
}

uint64_t random_signs() {
  const auto r = torch::randint(0, 1ll << 32, {2}, torch::kInt64);
  return static_cast<uint64_t>(r[0].item<int64_t>()) << 32 | static_cast<uint64_t>(r[1].item<int64_t>());
}

// x (R, C) bf16 or fp32 as a producer writes an MS-EDEN operand (quantize_mx's rows), finished
Nvfp4Tensor eden_rows(const torch::Tensor& x, uint64_t signs, uint64_t seed) {
  const auto t = empty_nvfp4(x.size(0), x.size(1), x.options(), false, false, seed, true, signs);
  kernels::MxOut out{};
  out.fp4 = nvfp4_out(t);
  quantize_mx_into(x, false, out, {});
  return nvfp4_finish(t);
}

constexpr float kEdenMul = 1.f / 8; // nvfp4.cuh's

} // namespace

// Each 64-group's <x.h, dequantized> is ||x.h||² within a scale step, and on average over seeds (as the simulation's
// EdenGroup64IsUnbiased); the tensor scale a power of two; the values scaled by 1/8
TEST(Nvfp4, EdenRowsAreUnbiased) {
  torch::manual_seed(0);
  at::globalContext().setFloat32MatmulPrecision("highest");
  const int64_t R = 128, C = 1024;
  const auto x = mild(R, C).to(torch::kBFloat16);
  const uint64_t signs = random_signs();
  const auto xh = torch::matmul(x.to(torch::kFloat32).view({R, C / 64, 64}), eden_rotation(signs));
  const auto norm2 = (xh * xh).sum(-1);
  auto mean = torch::zeros_like(norm2);
  const int n = 64;
  for (int i = 0; i < n; ++i) {
    const auto q = eden_rows(x, signs, static_cast<uint64_t>(i));
    const float k = std::log2(2688.f / q.amax.item<float>());
    EXPECT_EQ(k, std::round(k));
    const auto deq = nvfp4_to_bf16(q).to(torch::kFloat32).view({R, C / 64, 64}) / kEdenMul;
    const auto ratio = (xh * deq).sum(-1) / norm2;
    EXPECT_LT((ratio - 1).abs().max().item<float>(), 0.2) << i;
    mean += ratio / n;
  }
  EXPECT_LT((mean - 1).abs().max().item<float>(), 0.04);
  EXPECT_LT((mean - 1).mean().abs().item<float>(), 0.003);
}

// The GEMM of MS-EDEN operands: the simulation's error per estimate (64-groups), and estimates average towards the
// exact product
TEST(Nvfp4, EdenGemmAveragesToExact) {
  torch::manual_seed(0);
  at::globalContext().setFloat32MatmulPrecision("highest");
  const auto xa = mild(128, 1024).to(torch::kBFloat16), xb = mild(256, 1024) * 0.05;
  const auto exact = torch::mm(xa.to(torch::kFloat32), xb.t());
  auto sum = torch::zeros_like(exact);
  double real = 0, sim = 0;
  const int n = 64;
  for (int i = 0; i < n; ++i) {
    const uint64_t signs = random_signs();
    const auto est = nvfp4_gemm(eden_rows(xa, signs, 2 * i), eden_rows(xb, signs, 2 * i + 1)).to(torch::kFloat32);
    real += rel_err(est, exact) / n;
    sum += est;
    const auto h = nvfp4_eden_rotation(torch::kCUDA, 0, 64);
    const auto qa = nvfp4_eden(xa, h), qb = nvfp4_eden(xb, h);
    sim += rel_err(at::mm(qa.values, qb.values.t(), torch::kFloat32) * (qa.scale * qb.scale), exact) / n;
  }
  EXPECT_LT(real, 0.2);
  EXPECT_GT(real, 0.8 * sim);
  EXPECT_LT(real, 1.25 * sim);
  EXPECT_LT(rel_err(sum / n, exact), 0.25 * real);
}

// A weight's dgrad operand under eden_dgrad: its transpose MS-EDEN along out with the weight's signs (the gradient's
// producers get them from the weight), kept while the weight is unchanged
TEST(Nvfp4, EdenWeightTranspose) {
  torch::manual_seed(0);
  const Nvfp4Backward o{.wgrad = false, .dgrad = true, .eden_dgrad = true, .seed = 9};
  const auto w = mild(256, 512) * 0.05; // (out, in)
  Fp8WeightCache cache;
  cache.enabled = true;
  set_nvfp4_backward(&o);
  const auto q = quantize_fp8_weight(w, &cache, Fp8Recipe::Mx, true, false);
  const auto again = quantize_fp8_weight(w, &cache, Fp8Recipe::Mx, true, false);
  set_nvfp4_backward(nullptr);
  EXPECT_TRUE(q.fp4_t());
  EXPECT_FALSE(q.fp4());
  EXPECT_NE(q.eden_signs, 0);
  EXPECT_EQ(again.eden_signs, q.eden_signs);
  EXPECT_EQ(again.data_t.data_ptr(), q.data_t.data_ptr());
  // data_t (in, out) dequantized ~ w^T . rot / 8 per 64-group along out
  const auto xh = torch::matmul(
        w.t().contiguous().view({512, 4, 64}), eden_rotation(static_cast<uint64_t>(q.eden_signs)));
  const auto deq = nvfp4_to_bf16(q.nvfp4_t()).to(torch::kFloat32).view({512, 4, 64}) / kEdenMul;
  const auto ratio = (xh * deq).sum(-1) / (xh * xh).sum(-1);
  EXPECT_LT((ratio - 1).abs().max().item<float>(), 0.2);
  EXPECT_LT(rel_err(deq, xh), 0.3);
  // without MS-EDEN: 1D blocks along out, round to nearest, no signs
  const Nvfp4Backward sr{.wgrad = false, .dgrad = true};
  set_nvfp4_backward(&sr);
  const auto p = quantize_fp8_weight(w, &cache, Fp8Recipe::Mx, true, false);
  set_nvfp4_backward(nullptr);
  EXPECT_TRUE(p.fp4_t());
  EXPECT_EQ(p.eden_signs, 0);
  EXPECT_LT(rel_err(nvfp4_to_bf16(p.nvfp4_t()).to(torch::kFloat32), w.t()), 0.3);
}

// eden_dgrad: every dgrad GEMM but lm_head's (upstream of all: the last c_proj's weight gradient keeps its bits
// without NVFP4 wgrad) with both GEMM backends (CUTLASS: relu^2's fused dgrad), and with the NVFP4 forward
TEST(Nvfp4, EdenDgradInGpt) {
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
  {
    torch::NoGradGuard no_grad;
    for (auto& p : model->transformer->h->parameters())
      if (p.dim() == 2)
        p.normal_(0, 0.05);
  }
  model->set_fused(true);
  model->set_fp8_recipe(Fp8Recipe::Mx);
  model->set_fp8(true);
  model->set_attention(Attention::MX);
  model->set_loss_chunk_rows(512);
  const auto idx = torch::randint(0, 1000, {2, 256}, kCuda.dtype(torch::kInt64));
  auto grads = [&](const Nvfp4Backward* o) {
    set_nvfp4_backward(o);
    model->zero_grad();
    model->forward(idx, idx.roll(-1, 1)).backward();
    set_nvfp4_backward(nullptr);
    std::map<std::string, torch::Tensor> g;
    for (const auto& p : model->named_parameters(true))
      g[p.key()] = p.value().grad().clone();
    return g;
  };
  const auto mx = grads(nullptr);
  const std::array names{
        "transformer.h.0.attn.c_q.weight", "transformer.h.1.mlp.c_fc.weight", "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.0.attn.c_proj.weight", "transformer.wte.weight"};
  for (const bool cutlass : {false, true}) {
    std::optional<CutlassScope> scope;
    if (cutlass)
      scope.emplace();
    const Nvfp4Backward sr{.wgrad = false, .dgrad = true};
    const auto d = grads(&sr);
    for (const bool wgrad : {false, true}) {
      const Nvfp4Backward o{.wgrad = wgrad, .dgrad = true, .eden_dgrad = true, .seed = 5};
      const auto g = grads(&o);
      EXPECT_TRUE(torch::equal(g.at("lm_head.weight"), mx.at("lm_head.weight")));
      EXPECT_EQ(
            torch::equal(g.at("transformer.h.1.mlp.c_proj.weight"), mx.at("transformer.h.1.mlp.c_proj.weight")),
            !wgrad);
      for (const auto* name : names) {
        EXPECT_FALSE(torch::equal(g.at(name), mx.at(name))) << name << " cutlass " << cutlass << " wgrad " << wgrad;
        EXPECT_FALSE(torch::equal(g.at(name), d.at(name))) << name << " cutlass " << cutlass << " wgrad " << wgrad;
        EXPECT_LT(rel_err(g.at(name), mx.at(name)), 0.4) << name << " cutlass " << cutlass << " wgrad " << wgrad;
      }
    }
  }
  // with the NVFP4 forward (its weights' transposes MS-EDEN, not the forward's 16x16 blocks)
  const CutlassScope scope;
  model->set_nvfp4_fwd(true);
  const Nvfp4Backward o{.dgrad = true, .eden_dgrad = true};
  const auto g = grads(&o);
  model->set_nvfp4_fwd(false);
  for (const auto* name : names) {
    EXPECT_TRUE(g.at(name).isfinite().all().item<bool>()) << name;
    EXPECT_LT(rel_err(g.at(name), mx.at(name)), 0.7) << name;
  }
}

namespace {

// mean |a - b| per 16x16 block (rows / 16, cols / 16)
torch::Tensor block_mae(const torch::Tensor& a, const torch::Tensor& b) {
  const int64_t R = a.size(0), C = a.size(1);
  return (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs().view({R / 16, 16, C / 16, 16}).mean({1, 3});
}

} // namespace

// 4/6 weights: each 16x16 block's scale maps its max to 6 or to 4, never worse (sum |error|) than 6, better for some;
// the transpose shares the choice
TEST(Nvfp4, FourSixWeightsLowerBlockError) {
  torch::manual_seed(0);
  const int64_t R = 256, C = 384;
  const auto w = (spread(C, R).t().contiguous() * 0.05).to(torch::kBFloat16);
  const auto plain = empty_nvfp4(R, C, kCuda, false, false);
  quantize_nvfp4_2d(w, nvfp4_out(plain));
  const auto rows = empty_nvfp4(R, C, kCuda, false, false, 0, false, 0, true);
  const auto cols = empty_nvfp4(C, R, kCuda, false, false, 0, false, 0, true);
  quantize_nvfp4_2d(w, nvfp4_out(rows), nvfp4_out(cols));
  const auto [q, q_t] = nvfp4_finish(rows, cols);
  EXPECT_TRUE(torch::equal(nvfp4_to_bf16(q_t), nvfp4_to_bf16(q).t()));
  const auto e6 = block_mae(nvfp4_to_bf16(nvfp4_finish(plain)), w), e46 = block_mae(nvfp4_to_bf16(q), w);
  EXPECT_TRUE((e46 <= e6 * (1 + 1e-5)).all().item<bool>());
  EXPECT_GT((e46 < e6).sum().item<int64_t>(), 0); // Gaussian 256-value blocks: few (~2%) do better at 4
  EXPECT_LT(e46.sum().item<double>(), e6.sum().item<double>());
}

// 4/6 rows (Nvfp4Role::FwdInput under set_nvfp4_four_six): per 16-block, as the weights'; many more blocks pick 4
TEST(Nvfp4, FourSixRowsLowerBlockError) {
  torch::manual_seed(0);
  const auto x = spread(256, 512).to(torch::kBFloat16);
  const auto plain = quantize_mx(x, true, false, false, Nvfp4Role::FwdInput);
  set_nvfp4_four_six(true);
  const auto fs = quantize_mx(x, true, false, false, Nvfp4Role::FwdInput);
  set_nvfp4_four_six(false);
  const auto err = [&](const Fp8Tensor& t) {
    return (nvfp4_to_bf16(t.nvfp4()).to(torch::kFloat32) - x.to(torch::kFloat32)).abs().view({256, 32, 16}).mean(-1);
  };
  const auto e6 = err(plain), e46 = err(fs);
  // blocks over 2^13 below the tensor max: e4m3 subnormal scales, rounded coarser than the choice assumed
  const auto normal = x.to(torch::kFloat32).abs().view({256, 32, 16}).amax(-1) * 8192 >= fs.amax;
  EXPECT_GT(normal.to(torch::kFloat32).mean().item<double>(), 0.7);
  EXPECT_TRUE((e46.masked_select(normal) <= e6.masked_select(normal) * (1 + 1e-5)).all().item<bool>());
  EXPECT_GT((e46 < e6).to(torch::kFloat32).mean().item<double>(), 0.2); // Gaussian 16-blocks: ~1/3 pick 4
  EXPECT_LT(e46.sum().item<double>(), e6.sum().item<double>() * 0.97);
}
