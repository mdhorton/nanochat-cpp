// MXFP8 flash attention: quantizers bit-exact vs torch ops, and mx_attention_inputs' fused ones vs them; forward vs
// fp64 attention on the dequantized inputs (kernel bugs) and on the bf16 inputs (quantization error); forward +
// backward vs fp64 autograd.
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

double cosine(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64).flatten(), b64 = b.to(torch::kFloat64).flatten();
  return (a64.dot(b64) / (a64.norm() * b64.norm())).item<double>();
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

// along-token scales (..., 128) in dim order: dim d is stored at d % 8 * 16 + d / 16 * 2 + d / 8 % 2
torch::Tensor t_scale_dims(const torch::Tensor& scale) {
  std::vector<int64_t> pos(128);
  for (int64_t d = 0; d < 128; ++d)
    pos[d] = d % 8 * 16 + d / 16 * 2 + d / 8 % 2;
  return scale.index_select(-1, torch::tensor(pos, torch::TensorOptions().dtype(torch::kInt64)).to(scale.device()));
}

// (B, T, heads, 128) fp64 from mx_flash_quantize_t's output
torch::Tensor dequantize_t(const torch::Tensor& data, const torch::Tensor& scale) {
  const int64_t B = data.size(0), heads = data.size(1), T = data.size(3);
  const auto e = t_scale_dims(scale).transpose(2, 3);
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

TEST(MxFlash, QuantizeTMatchesTorchOps) {
  torch::manual_seed(0);
  const int64_t B = 2, T = 256, H = 2;
  const auto v = (torch::randn({B, T, H, 128}, kCuda) * torch::rand({B, 1, H, 128}, kCuda) * 10).to(torch::kBFloat16);
  const auto [data, scale] = mx_flash_quantize_t(v);
  const auto blocks = v.to(torch::kFloat32).permute({0, 2, 3, 1}).reshape({B, H, 128, T / 32, 32});
  const auto [want, e] = mx_reference(blocks);
  EXPECT_TRUE(torch::equal(data, want.view({B, H, 128, T}).index_select(3, token_order(T, false))));
  EXPECT_TRUE(torch::equal(t_scale_dims(scale), e.transpose(2, 3).to(torch::kUInt8)));
  // round trip
  EXPECT_LT(rel_norm(dequantize_t(data, scale), v), 0.05);
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
    const auto [vd, vs] = mx_flash_quantize_t(v);
    const auto [deq_out, deq_lse] = attention_ref(
          dequantize_rows(qd, qs), dequantize_rows(kd, ks), dequantize_t(vd, vs), c.window);
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

// MX forward + backward vs fp64 autograd, next to FA2's error; also vs the reference on MX-rounded (along head_dim)
// inputs, as S and dP see them. Errors are e4m3's (a few percent).
TEST(MxFlash, GradientsMatchReference) {
  torch::manual_seed(0);
  for (const auto& c : {Case{2, 256, 4, 2, 128}, Case{2, 2048, 4, 2, -1}, Case{2, 2048, 4, 2, 512}}) {
    const auto q0 = (1.2 * torch::randn({c.B, c.T, c.H, 128}, kCuda)).to(torch::kBFloat16);
    const auto k0 = (1.2 * torch::randn({c.B, c.T, c.Hkv, 128}, kCuda)).to(torch::kBFloat16);
    const auto v0 = torch::randn({c.B, c.T, c.Hkv, 128}, kCuda).to(torch::kBFloat16);
    const auto g = torch::randn({c.B, c.T, c.H, 128}, kCuda).to(torch::kBFloat16);
    // out, dq, dk, dv; kind 0: MX, 1: FA2, 2: fp64, 3: fp64 on MX-rounded inputs
    const auto run = [&](int kind) {
      const auto round = [&](const torch::Tensor& x) {
        return kind == 3 ? std::apply(dequantize_rows, mx_flash_quantize_rows(x)) : x.to(torch::kFloat64);
      };
      auto q = (kind < 2 ? q0.clone() : round(q0)).requires_grad_();
      auto k = (kind < 2 ? k0.clone() : round(k0)).requires_grad_();
      auto v = (kind < 2 ? v0.clone() : round(v0)).requires_grad_();
      torch::Tensor out;
      if (kind == 0)
        out = mx_flash_attention(q, k, v, c.window);
      else if (kind == 1) {
        const std::optional<int64_t> left = c.window >= 0 ? std::optional(c.window) : std::nullopt;
        out = std::get<0>(at::_flash_attention_forward(
              q, k, v, std::nullopt, std::nullopt, c.T, c.T, 0.0, true, false, std::nullopt, left,
              left ? std::optional<int64_t>(0) : std::nullopt));
      }
      else
        out = attention_ref(q, k, v, c.window).first;
      out.backward(kind == 3 ? round(g) : g.to(out.scalar_type()));
      return std::vector<torch::Tensor>{out, q.grad(), k.grad(), v.grad()};
    };
    const auto mx = run(0), fa2 = run(1), ref = run(2), ref_mx = run(3);
    const char* names[] = {"out", "dq", "dk", "dv"};
    std::cout << "B " << c.B << " T " << c.T << " H " << c.H << "/" << c.Hkv << " window " << c.window << ":";
    for (int i = 0; i < 4; ++i) {
      const double err = rel_norm(mx[i], ref[i]), cos = cosine(mx[i], ref[i]);
      std::cout << " " << names[i] << " " << err << " cos " << cos << " vs mx-rounded " << rel_norm(mx[i], ref_mx[i])
                << " (fa2 " << rel_norm(fa2[i], ref[i]) << ")";
      EXPECT_GT(cos, 0.99) << names[i];
    }
    std::cout << "\n";
  }
}

// mx_attention_inputs' quantize_attention: q, k, v unchanged; the MX inputs as the standalone quantizers write them;
// the attention on them identical to quantizing inside mx_flash_attention; backward runs.
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
  const char* names[] = {"q",  "q_scale",  "k",  "k_scale",  "v",  "v_scale",
                         "qt", "qt_scale", "kt", "kt_scale", "vt", "vt_scale"};
  for (const bool with_ve : {false, true}) {
    const auto inputs = [&](bool quantize) {
      return mx_attention_inputs(
            x, wq, wk, wv, cos, sin, 1.2, with_ve ? ve : torch::Tensor(), with_ve ? wg : torch::Tensor(), D, nullptr,
            nullptr, quantize);
    };
    const auto plain = inputs(false), fused = inputs(true);
    ASSERT_EQ(fused.size(), 15u);
    for (int i = 0; i < 3; ++i)
      EXPECT_TRUE(torch::equal(fused[i], plain[i])) << i << (with_ve ? " ve" : "");
    const auto q = fused[0].view({B, T, H, D}), k = fused[1].view({B, T, Hkv, D}), v = fused[2].view({B, T, Hkv, D});
    const auto want = mx_flash_quantize(q, k, v);
    const std::vector<torch::Tensor> w{want.q,  want.q_scale,  want.k,  want.k_scale,  want.v,  want.v_scale,
                                       want.qt, want.qt_scale, want.kt, want.kt_scale, want.vt, want.vt_scale};
    for (int i = 0; i < 12; ++i)
      EXPECT_TRUE(torch::equal(fused[3 + i], w[i])) << names[i] << (with_ve ? " ve" : "");

    const MxFlashInputs pre{fused[3], fused[4],  fused[5],  fused[6],  fused[7],  fused[8],
                            fused[9], fused[10], fused[11], fused[12], fused[13], fused[14]};
    const auto y = mx_flash_attention(q, k, v, 128, &pre);
    EXPECT_TRUE(torch::equal(y, mx_flash_forward(q, k, v, 128).first));
    y.to(torch::kFloat32).square().sum().backward();
    EXPECT_TRUE(x.grad().defined() && x.grad().isfinite().all().item<bool>());
    x.mutable_grad().reset();
  }
}
