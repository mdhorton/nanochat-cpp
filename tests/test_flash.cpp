// bf16 flash attention forward and bf16 / MXFP8 backward (every tile configuration) vs fp64 attention and FA2;
// gradients (FA2's backward) vs FA2 end to end; q, k, v as strided views of a merged qkv; MXFP8 quantization round
// trips.
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>

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

// dq, dk, dv of the fp64 reference for output gradient g
std::vector<torch::Tensor> ref_grads(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& g, int64_t window) {
  auto q64 = q.to(torch::kFloat64).requires_grad_(), k64 = k.to(torch::kFloat64).requires_grad_();
  auto v64 = v.to(torch::kFloat64).requires_grad_();
  attention_ref(q64, k64, v64, window).first.backward(g.to(torch::kFloat64));
  return {q64.grad(), k64.grad(), v64.grad()};
}

std::vector<torch::Tensor> fa2_grads(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& g, int64_t window) {
  auto qf = q.clone().requires_grad_(), kf = k.clone().requires_grad_(), vf = v.clone().requires_grad_();
  fa2(qf, kf, vf, window).backward(g);
  return {qf.grad(), kf.grad(), vf.grad()};
}

double cosine(const torch::Tensor& a, const torch::Tensor& b) {
  const auto a64 = a.to(torch::kFloat64).flatten(), b64 = b.to(torch::kFloat64).flatten();
  return (a64.dot(b64) / (a64.norm() * b64.norm())).item<double>();
}

// x (B, T, heads, 128) bf16 through flash_mx_quantize_rows and back (fp64)
torch::Tensor mx_rows_roundtrip(const torch::Tensor& x) {
  const int64_t B = x.size(0), T = x.size(1), heads = x.size(2);
  auto data = torch::empty({B, T, heads, 128}, kCuda.dtype(torch::kUInt8));
  auto scale = torch::empty({B, heads, T}, kCuda.dtype(torch::kInt32));
  kernels::flash_mx_quantize_rows(
        x.data_ptr(), x.stride(1), data.data_ptr(), static_cast<uint32_t*>(scale.data_ptr()), nullptr, nullptr, B, T,
        heads, at::cuda::getCurrentCUDAStream().stream());
  const auto e = scale.view(torch::kUInt8).view({B, heads, T, 4}).to(torch::kFloat64) - 127;
  return data.view(torch::kFloat8_e4m3fn).to(torch::kFloat64) *
         torch::pow(2.0, e).repeat_interleave(32, -1).transpose(1, 2);
}

// along-token scales (..., 128) in dim order: dim d is stored at d % 8 * 16 + d / 16 * 2 + d / 8 % 2
torch::Tensor t_scale_dims(const torch::Tensor& scale) {
  std::vector<int64_t> pos(128);
  for (int64_t d = 0; d < 128; ++d)
    pos[d] = d % 8 * 16 + d / 16 * 2 + d / 8 % 2;
  return scale.index_select(-1, torch::tensor(pos, torch::TensorOptions().dtype(torch::kInt64)).to(scale.device()));
}

const Case kBwdCases[] = {Case{2, 256, 2, 2, -1},  Case{2, 256, 2, 2, 128}, Case{1, 512, 4, 2, -1},
                          Case{1, 512, 4, 2, 200}, Case{1, 2048, 2, 2, -1}, Case{1, 2048, 2, 2, 512}};

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
  for (const auto& c : kBwdCases) {
    const auto q = (1.2 * torch::randn({c.B, c.T, c.H, 128}, kCuda)).to(torch::kBFloat16);
    const auto k = (1.2 * torch::randn({c.B, c.T, c.Hkv, 128}, kCuda)).to(torch::kBFloat16);
    const auto v = torch::randn({c.B, c.T, c.Hkv, 128}, kCuda).to(torch::kBFloat16);
    const auto g = torch::randn({c.B, c.T, c.H, 128}, kCuda).to(torch::kBFloat16);
    const auto want = ref_grads(q, k, v, g, c.window), theirs = fa2_grads(q, k, v, g, c.window);
    std::vector<double> fa2_err;
    for (int i = 0; i < 3; ++i)
      fa2_err.push_back(rel_norm(theirs[i], want[i]));
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

// MXFP8 backward (every tile configuration) vs fp64 autograd of the reference, next to FA2's error. Its error is
// e4m3's (3 mantissa bits, per 32 values): a few percent, not bf16's.
TEST(Flash, MxBackwardMatchesReference) {
  torch::manual_seed(0);
  for (const auto& c : kBwdCases) {
    const auto q = (1.2 * torch::randn({c.B, c.T, c.H, 128}, kCuda)).to(torch::kBFloat16);
    const auto k = (1.2 * torch::randn({c.B, c.T, c.Hkv, 128}, kCuda)).to(torch::kBFloat16);
    const auto v = torch::randn({c.B, c.T, c.Hkv, 128}, kCuda).to(torch::kBFloat16);
    const auto g = torch::randn({c.B, c.T, c.H, 128}, kCuda).to(torch::kBFloat16);
    const auto want = ref_grads(q, k, v, g, c.window), theirs = fa2_grads(q, k, v, g, c.window);
    // the reference on MX-rounded (along head_dim) inputs: S and dP as the kernels compute them
    const auto want_mx = ref_grads(
          mx_rows_roundtrip(q), mx_rows_roundtrip(k), mx_rows_roundtrip(v), mx_rows_roundtrip(g), c.window);
    const auto [out, lse] = flash_forward(q, k, v, c.window);
    for (int dqv = 0; dqv < kernels::kFlashBwdMxDqVariants; ++dqv) {
      for (int dkvv = 0; dkvv < kernels::kFlashBwdMxDkvVariants; ++dkvv) {
        const auto [dq, dk, dv] = flash_backward_mx(g, q, k, v, out, lse, c.window, dqv, dkvv);
        const std::vector<torch::Tensor> got{dq, dk, dv};
        std::cout << "B " << c.B << " T " << c.T << " H " << c.H << "/" << c.Hkv << " window " << c.window << " ["
                  << kernels::flash_bwd_mx_dq_variant_name(dqv) << "; " << kernels::flash_bwd_mx_dkv_variant_name(dkvv)
                  << "]:";
        const char* names[] = {"dq", "dk", "dv"};
        for (int i = 0; i < 3; ++i) {
          const double err = rel_norm(got[i], want[i]), cos = cosine(got[i], want[i]);
          const double err_mx = rel_norm(got[i], want_mx[i]);
          std::cout << " " << names[i] << " " << err << " cos " << cos << " vs mx-rounded " << err_mx << " (fa2 "
                    << rel_norm(theirs[i], want[i]) << ")";
          EXPECT_GT(cos, 0.99) << names[i];
        }
        std::cout << "\n";
      }
    }
  }
}

// flash_mx_quantize_rows / _t dequantized vs their bf16 input: e4m3's error, so any layout or permutation slip shows;
// flash_mx_quantize_dout identical to them
TEST(Flash, MxQuantizeRoundTrip) {
  torch::manual_seed(0);
  const int64_t B = 2, T = 256, heads = 3, D = 128;
  const auto x = torch::randn({B, T, heads + 1, D}, kCuda).to(torch::kBFloat16).narrow(2, 0, heads); // strided
  const auto o = torch::randn({B, T, heads, D}, kCuda).to(torch::kBFloat16);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto u8 = kCuda.dtype(torch::kUInt8);
  const auto pow2 = [](const torch::Tensor& e) {
    return torch::pow(2.0, e.to(torch::kFloat64) - 127);
  };
  const auto e4m3 = [](const torch::Tensor& d) {
    return d.view(torch::kFloat8_e4m3fn).to(torch::kFloat64);
  };
  const auto x64 = x.to(torch::kFloat64);

  auto data = torch::empty({B, T, heads, D}, u8), scale = torch::empty({B, heads, T}, kCuda.dtype(torch::kInt32));
  auto delta = torch::empty({B, heads, T}, kCuda.dtype(torch::kFloat32));
  kernels::flash_mx_quantize_rows(
        x.data_ptr(), x.stride(1), data.data_ptr(), static_cast<uint32_t*>(scale.data_ptr()), o.data_ptr(),
        delta.data_ptr<float>(), B, T, heads, stream);
  // scale bytes (B, heads, T, 4) -> per value (B, T, heads, D)
  const auto row_scale =
        pow2(scale.view(torch::kUInt8).view({B, heads, T, 4})).repeat_interleave(32, -1).transpose(1, 2);
  const double row_err = rel_norm(e4m3(data) * row_scale, x64);
  const double delta_err = rel_norm(delta, (x64 * o.to(torch::kFloat64)).sum(-1).transpose(1, 2));

  auto tdata = torch::empty({B, heads, D, T}, u8), tscale = torch::empty({B, heads, T / 32, D}, u8);
  kernels::flash_mx_quantize_t(
        x.data_ptr(), x.stride(1), tdata.data_ptr(), tscale.data_ptr<uint8_t>(), B, T, heads, stream);
  // position p of each 16 holds token perm[p]
  const int64_t perm[16] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
  std::vector<int64_t> pos(T);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t p = 0; p < 16; ++p)
      if (perm[p] == t % 16)
        pos[t] = t / 16 * 16 + p;
  const auto tokens = e4m3(tdata).index_select(-1, torch::tensor(pos, kCuda.dtype(torch::kInt64)));
  const auto t_scale = pow2(t_scale_dims(tscale).transpose(-1, -2)).repeat_interleave(32, -1); // (B, heads, D, T)
  const double t_err = rel_norm(tokens * t_scale, x64.permute({0, 2, 3, 1}));
  std::cout << "rows " << row_err << ", transposed " << t_err << ", delta " << delta_err << "\n";
  EXPECT_LT(row_err, 0.04);
  EXPECT_LT(t_err, 0.04);
  EXPECT_LT(delta_err, 1e-5);

  // flash_mx_quantize_dout: both of the above in one pass
  const auto xc = x.contiguous();
  auto ddata = torch::empty_like(data), dscale = torch::empty_like(scale), ddelta = torch::empty_like(delta);
  auto dtdata = torch::empty_like(tdata), dtscale = torch::empty_like(tscale);
  kernels::flash_mx_quantize_dout(
        xc.data_ptr(), o.data_ptr(), ddata.data_ptr(), static_cast<uint32_t*>(dscale.data_ptr()), dtdata.data_ptr(),
        dtscale.data_ptr<uint8_t>(), ddelta.data_ptr<float>(), B, T, heads, stream);
  EXPECT_TRUE(torch::equal(ddata, data));
  EXPECT_TRUE(torch::equal(dscale, scale));
  EXPECT_TRUE(torch::equal(ddelta, delta));
  EXPECT_TRUE(torch::equal(dtdata, tdata));
  EXPECT_TRUE(torch::equal(dtscale, tscale));
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
