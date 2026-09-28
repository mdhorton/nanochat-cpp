// bf16 flash attention forward (every tile configuration) vs fp64 attention and FA2; gradients (FA2's backward) vs FA2
// end to end; q, k, v as strided views of a merged qkv.
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "nanochat/model/flash.h"
#include "nanochat/model/flash_kernel.h"

using namespace nanochat;

namespace {

const auto kCuda = torch::TensorOptions().device(torch::kCUDA);

// ||a - b|| / ||b||
double rel_norm(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).norm() / b64.norm()).item<double>();
}

// fp64 causal attention with a left window; returns out (B, T, H, D) and lse (B, H, T)
std::pair<torch::Tensor, torch::Tensor> attention_ref(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window) {
  const int64_t T = q.size(1), H = q.size(2), D = q.size(3), rep = H / k.size(2);
  const auto qt = q.to(torch::kFloat64).transpose(1, 2);
  const auto kt = k.to(torch::kFloat64).transpose(1, 2).repeat_interleave(rep, 1);
  const auto vt = v.to(torch::kFloat64).transpose(1, 2).repeat_interleave(rep, 1);
  auto s = torch::matmul(qt, kt.transpose(-1, -2)) / std::sqrt(static_cast<double>(D));
  const auto opts = kCuda.dtype(torch::kInt64);
  const auto row = torch::arange(T, opts).unsqueeze(1), col = torch::arange(T, opts).unsqueeze(0);
  auto keep = col <= row;
  if (window >= 0)
    keep = keep & ((row - col) <= window);
  s = s.masked_fill(keep.logical_not(), -INFINITY);
  const auto lse = s.logsumexp(-1);
  return {torch::matmul((s - lse.unsqueeze(-1)).exp(), vt).transpose(1, 2), lse};
}

torch::Tensor fa2(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window) {
  const int64_t T = q.size(1);
  const std::optional<int64_t> left = window >= 0 ? std::optional(window) : std::nullopt;
  return std::get<0>(at::_flash_attention_forward(
        q, k, v, std::nullopt, std::nullopt, T, T, 0.0, true, false, std::nullopt, left,
        left ? std::optional<int64_t>(0) : std::nullopt));
}

struct Case {
  int64_t B, T, H, Hkv, window;
};

} // namespace

TEST(Flash, ForwardMatchesReference) {
  torch::manual_seed(0);
  for (const auto& c :
       {Case{2, 256, 2, 2, -1}, Case{2, 256, 2, 2, 128}, Case{1, 512, 4, 2, -1}, Case{1, 512, 4, 2, 200},
        Case{2, 2048, 2, 2, -1}, Case{2, 2048, 2, 2, 512}}) {
    // q, k as after the QK norm (rms 1.2)
    const auto q = (1.2 * torch::randn({c.B, c.T, c.H, 128}, kCuda)).to(torch::kBFloat16);
    const auto k = (1.2 * torch::randn({c.B, c.T, c.Hkv, 128}, kCuda)).to(torch::kBFloat16);
    const auto v = torch::randn({c.B, c.T, c.Hkv, 128}, kCuda).to(torch::kBFloat16);
    const auto [ref_out, ref_lse] = attention_ref(q, k, v, c.window);
    const double fa2_err = rel_norm(fa2(q, k, v, c.window), ref_out);
    for (int variant = 0; variant < kernels::kFlashVariants; ++variant) {
      const auto [out, lse] = flash_forward(q, k, v, c.window, variant);
      const double out_err = rel_norm(out, ref_out);
      const double lse_err = (lse.to(torch::kFloat64) - ref_lse).abs().max().item<double>();
      std::cout << "B " << c.B << " T " << c.T << " H " << c.H << "/" << c.Hkv << " window " << c.window << " ["
                << kernels::flash_variant_name(variant) << "]: out " << out_err << " (fa2 " << fa2_err << "), lse "
                << lse_err << "\n";
      // P is rounded to bf16 before P·V, as FA2
      EXPECT_LT(out_err, 2 * fa2_err + 1e-3) << kernels::flash_variant_name(variant);
      EXPECT_LT(lse_err, 1e-4) << kernels::flash_variant_name(variant);
    }
  }
}

// the backward (every tile configuration) vs fp64 autograd of the reference, next to FA2's error
TEST(Flash, BackwardMatchesReference) {
  torch::manual_seed(0);
  for (const auto& c :
       {Case{2, 256, 2, 2, -1}, Case{2, 256, 2, 2, 128}, Case{1, 512, 4, 2, -1}, Case{1, 512, 4, 2, 200},
        Case{1, 2048, 2, 2, -1}, Case{1, 2048, 2, 2, 512}}) {
    const auto q = (1.2 * torch::randn({c.B, c.T, c.H, 128}, kCuda)).to(torch::kBFloat16);
    const auto k = (1.2 * torch::randn({c.B, c.T, c.Hkv, 128}, kCuda)).to(torch::kBFloat16);
    const auto v = torch::randn({c.B, c.T, c.Hkv, 128}, kCuda).to(torch::kBFloat16);
    const auto g = torch::randn({c.B, c.T, c.H, 128}, kCuda).to(torch::kBFloat16);
    std::vector<torch::Tensor> want;
    {
      auto q64 = q.to(torch::kFloat64).requires_grad_(), k64 = k.to(torch::kFloat64).requires_grad_();
      auto v64 = v.to(torch::kFloat64).requires_grad_();
      attention_ref(q64, k64, v64, c.window).first.backward(g.to(torch::kFloat64));
      want = {q64.grad(), k64.grad(), v64.grad()};
    }
    std::vector<double> fa2_err;
    {
      auto qf = q.clone().requires_grad_(), kf = k.clone().requires_grad_(), vf = v.clone().requires_grad_();
      fa2(qf, kf, vf, c.window).backward(g);
      fa2_err = {rel_norm(qf.grad(), want[0]), rel_norm(kf.grad(), want[1]), rel_norm(vf.grad(), want[2])};
    }
    const auto [out, lse] = flash_forward(q, k, v, c.window);
    for (int dqv = 0; dqv < kernels::kFlashBwdDqVariants; ++dqv) {
      for (int dkvv = 0; dkvv < kernels::kFlashBwdDkvVariants; ++dkvv) {
        const auto [dq, dk, dv] = flash_backward(g, q, k, v, out, lse, c.window, dqv, dkvv);
        const std::vector<torch::Tensor> got{dq, dk, dv};
        std::cout << "B " << c.B << " T " << c.T << " H " << c.H << "/" << c.Hkv << " window " << c.window << " ["
                  << kernels::flash_bwd_dq_variant_name(dqv) << "; " << kernels::flash_bwd_dkv_variant_name(dkvv)
                  << "]:";
        const char* names[] = {"dq", "dk", "dv"};
        for (int i = 0; i < 3; ++i) {
          const double err = rel_norm(got[i], want[i]);
          std::cout << " " << names[i] << " " << err << " (fa2 " << fa2_err[i] << ")";
          EXPECT_LT(err, 2 * fa2_err[i] + 1e-3) << names[i];
        }
        std::cout << "\n";
      }
    }
  }
}

// flash forward + backward vs FA2 forward + backward
TEST(Flash, GradientsMatchFa2) {
  torch::manual_seed(0);
  for (const int64_t window : {int64_t{-1}, int64_t{512}}) {
    const int64_t B = 2, T = 2048, H = 4, Hkv = 2;
    const auto q0 = (1.2 * torch::randn({B, T, H, 128}, kCuda)).to(torch::kBFloat16);
    const auto k0 = (1.2 * torch::randn({B, T, Hkv, 128}, kCuda)).to(torch::kBFloat16);
    const auto v0 = torch::randn({B, T, Hkv, 128}, kCuda).to(torch::kBFloat16);
    const auto g = torch::randn({B, T, H, 128}, kCuda).to(torch::kBFloat16);
    const auto run = [&](bool ours) {
      auto q = q0.clone().requires_grad_(), k = k0.clone().requires_grad_(), v = v0.clone().requires_grad_();
      const auto out = ours ? flash_attention(q, k, v, window) : fa2(q, k, v, window);
      out.backward(g);
      return std::vector<torch::Tensor>{out, q.grad(), k.grad(), v.grad()};
    };
    const auto ours = run(true), theirs = run(false);
    const char* names[] = {"out", "dq", "dk", "dv"};
    for (int i = 0; i < 4; ++i) {
      const double d = rel_norm(ours[i], theirs[i]);
      std::cout << "window " << window << " " << names[i] << " flash vs fa2 " << d << "\n";
      EXPECT_LT(d, 1e-2) << names[i];
    }
  }
}

// q, k, v as views of one (B, T, (H + 2 Hkv) * D) buffer, as the model passes them
TEST(Flash, StridedViews) {
  torch::manual_seed(0);
  const int64_t B = 2, T = 256, H = 4, Hkv = 2, D = 128;
  const auto qkv = torch::randn({B, T, (H + 2 * Hkv) * D}, kCuda).to(torch::kBFloat16);
  const auto q = qkv.narrow(2, 0, H * D).view({B, T, H, D});
  const auto k = qkv.narrow(2, H * D, Hkv * D).view({B, T, Hkv, D});
  const auto v = qkv.narrow(2, (H + Hkv) * D, Hkv * D).view({B, T, Hkv, D});
  const auto [out, lse] = flash_forward(q, k, v, 100);
  const auto [out_c, lse_c] = flash_forward(q.contiguous(), k.contiguous(), v.contiguous(), 100);
  EXPECT_TRUE(torch::equal(out, out_c));
  EXPECT_TRUE(torch::equal(lse, lse_c));
}
