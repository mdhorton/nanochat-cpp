// MXFP8 flash attention: quantizers bit-exact vs torch ops, and mx_attention_inputs' fused ones vs them; forward vs
// fp64 attention on the dequantized inputs (kernel bugs) and on the bf16 inputs (quantization error); gradients (FA2's
// backward) vs FA2 end to end.
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "nanochat/model/mx_attention.h"
#include "nanochat/model/mx_flash.h"

using namespace nanochat;

namespace {

const auto kCuda = torch::TensorOptions().device(torch::kCUDA);

// ||a - b|| / ||b||
double rel_norm(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64), b64 = b.to(torch::kFloat64);
  return ((a64 - b64).norm() / b64.norm()).item<double>();
}

// MXFP8 exponent (mx_kernel.cuh's mx_exponent) of each block's amax, int32
torch::Tensor mx_exponent(const torch::Tensor& amax) {
  const auto bits = amax.to(torch::kFloat32).contiguous().view(torch::kInt32);
  const auto e = bits.bitwise_right_shift(23) - 8 + (bits.bitwise_and(0x7fffff) > 0x600000).to(torch::kInt32);
  return e.clamp(0, 253);
}

// blocks (..., 32) fp32 -> e4m3 bytes and exponents
std::pair<torch::Tensor, torch::Tensor> mx_reference(const torch::Tensor& blocks) {
  const auto e = mx_exponent(blocks.abs().amax(-1));
  const auto mul = torch::pow(2.0, (127 - e).to(torch::kFloat64)).to(torch::kFloat32).unsqueeze(-1);
  return {(blocks * mul).to(torch::kFloat8_e4m3fn).view(torch::kUInt8), e};
}

// position p of each 16 holds token perm16[p]
torch::Tensor token_order(int64_t T, bool inverse) {
  std::vector<int64_t> idx(T);
  for (int64_t p = 0; p < T; ++p) {
    const int64_t i = p % 16, t = p / 16 * 16 + 2 * (i / 4) + i % 2 + 8 * (i % 4 / 2);
    if (inverse)
      idx[t] = p;
    else
      idx[p] = t;
  }
  return torch::tensor(idx, torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA);
}

torch::Tensor pow2(const torch::Tensor& e) {
  return torch::pow(2.0, (e.to(torch::kInt32) - 127).to(torch::kFloat64));
}

// (B, T, heads, 128) fp64 from mx_flash_quantize_rows' output
torch::Tensor dequantize_rows(const torch::Tensor& data, const torch::Tensor& scale) {
  const int64_t B = data.size(0), T = data.size(1), heads = data.size(2);
  const auto e = scale.view(torch::kUInt8).view({B, heads, T, 4}).permute({0, 2, 1, 3});
  const auto x = data.view(torch::kFloat8_e4m3fn).to(torch::kFloat64).view({B, T, heads, 4, 32});
  return (x * pow2(e).unsqueeze(-1)).view({B, T, heads, 128});
}

// (B, T, heads, 128) fp64 from mx_flash_quantize_vt's output
torch::Tensor dequantize_vt(const torch::Tensor& data, const torch::Tensor& scale) {
  const int64_t B = data.size(0), heads = data.size(1), T = data.size(3);
  const auto e = scale.permute({0, 1, 3, 2, 4}).reshape({B, heads, 128, T / 32});
  const auto x = data.index_select(3, token_order(T, true)).view(torch::kFloat8_e4m3fn).to(torch::kFloat64);
  return (x.view({B, heads, 128, T / 32, 32}) * pow2(e).unsqueeze(-1)).view({B, heads, 128, T}).permute({0, 3, 1, 2});
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

struct Case {
  int64_t B, T, H, Hkv, window;
};

} // namespace

TEST(MxFlash, QuantizeRowsMatchesTorchOps) {
  torch::manual_seed(0);
  const int64_t B = 2, T = 256, H = 3;
  auto x = (torch::randn({B, T, H, 128}, kCuda) * torch::rand({B, T, H, 1}, kCuda) * 10).to(torch::kBFloat16);
  x.index_put_({0, 0, 0}, 0); // amax 0
  const auto [data, scale] = mx_flash_quantize_rows(x);
  const auto [want, e] = mx_reference(x.to(torch::kFloat32).view({B, T, H, 4, 32}));
  EXPECT_TRUE(torch::equal(data.view({B, T, H, 4, 32}), want));
  EXPECT_TRUE(torch::equal(scale.view(torch::kUInt8).view({B, H, T, 4}), e.permute({0, 2, 1, 3}).to(torch::kUInt8)));

  // q as a strided view of a merged qkv buffer
  const auto qkv = torch::randn({B, T, 3 * H * 128}, kCuda).to(torch::kBFloat16);
  const auto q = qkv.narrow(2, H * 128, H * 128).view({B, T, H, 128});
  const auto [qd, qs] = mx_flash_quantize_rows(q);
  const auto [qd_contig, qs_contig] = mx_flash_quantize_rows(q.contiguous());
  EXPECT_TRUE(torch::equal(qd, qd_contig));
  EXPECT_TRUE(torch::equal(qs, qs_contig));
}

TEST(MxFlash, QuantizeVtMatchesTorchOps) {
  torch::manual_seed(0);
  const int64_t B = 2, T = 256, H = 2;
  const auto v = (torch::randn({B, T, H, 128}, kCuda) * torch::rand({B, 1, H, 128}, kCuda) * 10).to(torch::kBFloat16);
  const auto [data, scale] = mx_flash_quantize_vt(v);
  const auto blocks = v.to(torch::kFloat32).permute({0, 2, 3, 1}).reshape({B, H, 128, T / 32, 32});
  const auto [want, e] = mx_reference(blocks);
  EXPECT_TRUE(torch::equal(data, want.view({B, H, 128, T}).index_select(3, token_order(T, false))));
  EXPECT_TRUE(torch::equal(scale, e.view({B, H, 128, T / 64, 2}).permute({0, 1, 3, 2, 4}).to(torch::kUInt8)));
  // round trip
  EXPECT_LT(rel_norm(dequantize_vt(data, scale), v), 0.05);
}

TEST(MxFlash, ForwardMatchesReference) {
  torch::manual_seed(0);
  for (const auto& c :
       {Case{2, 256, 2, 2, -1}, Case{2, 256, 2, 2, 128}, Case{1, 512, 4, 2, -1}, Case{1, 512, 4, 2, 200},
        Case{2, 2048, 2, 2, -1}, Case{2, 2048, 2, 2, 512}}) {
    // q, k as after the QK norm (rms 1.2)
    const auto q = (1.2 * torch::randn({c.B, c.T, c.H, 128}, kCuda)).to(torch::kBFloat16);
    const auto k = (1.2 * torch::randn({c.B, c.T, c.Hkv, 128}, kCuda)).to(torch::kBFloat16);
    const auto v = torch::randn({c.B, c.T, c.Hkv, 128}, kCuda).to(torch::kBFloat16);
    const auto [out, lse] = mx_flash_forward(q, k, v, c.window);

    const auto [qd, qs] = mx_flash_quantize_rows(q);
    const auto [kd, ks] = mx_flash_quantize_rows(k);
    const auto [vd, vs] = mx_flash_quantize_vt(v);
    const auto [deq_out, deq_lse] = attention_ref(
          dequantize_rows(qd, qs), dequantize_rows(kd, ks), dequantize_vt(vd, vs), c.window);
    const auto [ref_out, ref_lse] = attention_ref(q, k, v, c.window);
    const double out_deq = rel_norm(out, deq_out), out_ref = rel_norm(out, ref_out);
    const double lse_deq = (lse.to(torch::kFloat64) - deq_lse).abs().max().item<double>();
    const double lse_ref = (lse.to(torch::kFloat64) - ref_lse).abs().max().item<double>();
    std::cout << "B " << c.B << " T " << c.T << " H " << c.H << "/" << c.Hkv << " window " << c.window
              << ": out vs dequantized " << out_deq << ", vs bf16 " << out_ref << "; lse vs dequantized " << lse_deq
              << ", vs bf16 " << lse_ref << "\n";
    // vs dequantized: P's e4m3 rounding only (~1.7%); vs bf16: plus Q, K, V's (~5.5%, lse ~0.1 at worst)
    EXPECT_LT(out_deq, 0.03);
    EXPECT_LT(out_ref, 0.08);
    EXPECT_LT(lse_deq, 1e-4);
    EXPECT_LT(lse_ref, 0.25);
  }
}

// MX forward + FA2 backward vs FA2 forward + backward
TEST(MxFlash, GradientsMatchFa2) {
  torch::manual_seed(0);
  for (const int64_t window : {int64_t{-1}, int64_t{512}}) {
    const int64_t B = 2, T = 2048, H = 4, Hkv = 2;
    const auto q0 = (1.2 * torch::randn({B, T, H, 128}, kCuda)).to(torch::kBFloat16);
    const auto k0 = (1.2 * torch::randn({B, T, Hkv, 128}, kCuda)).to(torch::kBFloat16);
    const auto v0 = torch::randn({B, T, Hkv, 128}, kCuda).to(torch::kBFloat16);
    const auto g = torch::randn({B, T, H, 128}, kCuda).to(torch::kBFloat16);
    const auto run = [&](bool mx) {
      auto q = q0.clone().requires_grad_(), k = k0.clone().requires_grad_(), v = v0.clone().requires_grad_();
      torch::Tensor out;
      if (mx)
        out = mx_flash_attention(q, k, v, window);
      else {
        const std::optional<int64_t> left = window >= 0 ? std::optional(window) : std::nullopt;
        out = std::get<0>(at::_flash_attention_forward(
              q, k, v, std::nullopt, std::nullopt, T, T, 0.0, true, false, std::nullopt, left,
              left ? std::optional<int64_t>(0) : std::nullopt));
      }
      out.backward(g);
      return std::vector<torch::Tensor>{out, q.grad(), k.grad(), v.grad()};
    };
    const auto mx = run(true), fa2 = run(false);
    const char* names[] = {"out", "dq", "dk", "dv"};
    for (int i = 0; i < 4; ++i) {
      const double d = rel_norm(mx[i], fa2[i]);
      std::cout << "window " << window << " " << names[i] << " mx vs fa2 " << d << "\n";
      EXPECT_LT(d, 0.1) << names[i];
    }
  }
}

// mx_attention_inputs' quantize_attention: q, k, v unchanged; the MX inputs as the standalone quantizers write them;
// the attention on them identical to quantizing inside mx_flash_forward; backward runs.
TEST(MxFlash, FusedQuantizationMatchesStandalone) {
  torch::manual_seed(0);
  const int64_t B = 2, T = 256, N = B * T, C = 256, D = 128, H = 4, Hkv = 2, gate_cols = 12;
  const auto freqs = torch::outer(torch::arange(T, kCuda), torch::rand({D / 2}, kCuda));
  const auto cos = freqs.cos().to(torch::kBFloat16).view({1, T, 1, D / 2});
  const auto sin = freqs.sin().to(torch::kBFloat16).view({1, T, 1, D / 2});
  const auto x = torch::randn({N, C}, kCuda).to(torch::kBFloat16).requires_grad_();
  const auto wq = (torch::randn({H * D, C}, kCuda) * 0.05).requires_grad_();
  const auto wk = (torch::randn({Hkv * D, C}, kCuda) * 0.07).requires_grad_();
  const auto wv = (torch::randn({Hkv * D, C}, kCuda) * 0.09).requires_grad_();
  const auto wg = (torch::randn({Hkv, gate_cols}, kCuda) * 0.3).requires_grad_();
  const auto ve = torch::randn({N, Hkv * D}, kCuda).to(torch::kBFloat16).requires_grad_();
  for (const bool with_ve : {false, true}) {
    const auto inputs = [&](bool quantize) {
      return mx_attention_inputs(
            x, wq, wk, wv, cos, sin, 1.2, with_ve ? ve : torch::Tensor(), with_ve ? wg : torch::Tensor(), D, nullptr,
            nullptr, quantize);
    };
    const auto plain = inputs(false), fused = inputs(true);
    ASSERT_EQ(fused.size(), 9u);
    for (int i = 0; i < 3; ++i)
      EXPECT_TRUE(torch::equal(fused[i], plain[i])) << i << (with_ve ? " ve" : "");
    const auto q = fused[0].view({B, T, H, D}), k = fused[1].view({B, T, Hkv, D}), v = fused[2].view({B, T, Hkv, D});
    const auto [qd, qs] = mx_flash_quantize_rows(q);
    const auto [kd, ks] = mx_flash_quantize_rows(k);
    const auto [vd, vs] = mx_flash_quantize_vt(v);
    const std::vector<torch::Tensor> want{qd, qs, kd, ks, vd, vs};
    for (int i = 0; i < 6; ++i)
      EXPECT_TRUE(torch::equal(fused[3 + i], want[i])) << "mx input " << i << (with_ve ? " ve" : "");

    const MxFlashInputs pre{fused[3], fused[4], fused[5], fused[6], fused[7], fused[8]};
    const auto y = mx_flash_attention(q, k, v, 128, &pre);
    EXPECT_TRUE(torch::equal(y, mx_flash_forward(q, k, v, 128).first));
    y.to(torch::kFloat32).square().sum().backward();
    EXPECT_TRUE(x.grad().defined() && x.grad().isfinite().all().item<bool>());
    x.mutable_grad().reset();
  }
}
