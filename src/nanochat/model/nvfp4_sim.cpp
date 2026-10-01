#include "nanochat/model/nvfp4_sim.h"

#include <atomic>
#include <cmath>
#include <map>
#include <mutex>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include "nanochat/model/fp8.h"
#include "nanochat/model/mx_gemm.h"
#include "nanochat/model/nvfp4.h"
#include "nanochat/model/nvfp4_sim_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

std::atomic<uint64_t> seed_counter{1};

// 2D (N, in) x (out, in) -> (N, out)
torch::Tensor mx_matmul(const torch::Tensor& a, const torch::Tensor& b, torch::ScalarType dtype) {
  const auto aq = quantize_mx(a, true, false), bq = quantize_mx(b, true, false);
  return mx_gemm(aq.data, aq.inv_scale, bq.data, bq.inv_scale, dtype);
}

// splitmix64's finalizer
uint64_t mix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// 2D (M, K) x (N, K) -> (M, N) fp32, with MS-EDEN operands sharing one rotation along K. Signs fresh per call, or
// (eden_fixed_signs) from the step, the Linear (id) and the GEMM (kind).
torch::Tensor eden_matmul(const torch::Tensor& a, const torch::Tensor& b, const Nvfp4Options& o, int64_t id, int kind) {
  const auto h = o.eden_fixed_signs ? nvfp4_eden_rotation(
                                            a.device(), mix64((o.seed << 40) + (o.step << 16) + (id << 2) + kind),
                                            o.eden_group, false)
                                    : nvfp4_eden_rotation(a.device(), o.seed, o.eden_group);
  const auto qa = nvfp4_eden(a, h, o.seed), qb = nvfp4_eden(b, h, o.seed);
  return at::mm(qa.values, qb.values.t(), torch::kFloat32) * (qa.scale * qb.scale);
}

// |a| -> nearest FP4 magnitude (saturating), ties to the even mantissa: each threshold overrides the ones above
torch::Tensor e2m1_rn(const torch::Tensor& a) {
  auto q = torch::full_like(a, 6.f);
  q = torch::where(a <= 5, 4.f, q);
  q = torch::where(a < 3.5, 3.f, q);
  q = torch::where(a <= 2.5, 2.f, q);
  q = torch::where(a < 1.75, 1.5f, q);
  q = torch::where(a <= 1.25, 1.f, q);
  q = torch::where(a < .75, .5f, q);
  return torch::where(a <= .25, 0.f, q);
}

// the e4m3 value `step` codes from positive e4m3 codes, within [0, 448]
torch::Tensor e4m3_step(const torch::Tensor& bits, int step) {
  return (bits.to(torch::kInt16) + step)
        .clamp(0, 0x7e)
        .to(torch::kUInt8)
        .view(torch::kFloat8_e4m3fn)
        .to(torch::kFloat32);
}

class Nvfp4SimMatmul : public torch::autograd::Function<Nvfp4SimMatmul> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& input, const torch::Tensor& weight, const Nvfp4Options* o,
        int64_t id, bool eden) {
    ctx->save_for_backward({input, weight});
    ctx->saved_data["o"] = reinterpret_cast<int64_t>(o);
    ctx->saved_data["id"] = id;
    ctx->saved_data["eden"] = eden;
    if (!o->fwd)
      return mx_matmul(input, weight, input.scalar_type());
    if (o->rht_fwd)
      return torch::mm(nvfp4_fake_quant(nvfp4_rht(input)), nvfp4_fake_quant(nvfp4_rht(weight)).t());
    return torch::mm(nvfp4_fake_quant(input), nvfp4_fake_quant(weight, false, o->weight_2d).t());
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    const auto s = ctx->get_saved_variables();
    const auto &input = s[0], &weight = s[1];
    const auto* o = reinterpret_cast<const Nvfp4Options*>(ctx->saved_data["o"].toInt());
    const int64_t id = ctx->saved_data["id"].toInt();
    const bool eden = ctx->saved_data["eden"].toBool();
    const auto go = grad_outputs[0].contiguous();
    const auto dtype = go.scalar_type();
    // dX = go . W: contraction over out
    torch::Tensor dx;
    if (!o->dgrad)
      dx = mx_matmul(go, weight.t().contiguous(), dtype);
    else if (o->eden_dgrad && eden)
      dx = eden_matmul(go, weight.t().contiguous(), *o, id, 1).to(dtype);
    else {
      // W^T (in, out), quantized along out
      torch::Tensor w_t;
      if (o->rht_dgrad)
        w_t = nvfp4_fake_quant(nvfp4_rht(weight.t().contiguous()));
      else if (o->weight_2d)
        w_t = nvfp4_fake_quant(weight, false, true).t();
      else
        w_t = nvfp4_fake_quant(weight.t().contiguous());
      const auto g = nvfp4_fake_quant(o->rht_dgrad ? nvfp4_rht(go) : go, o->sr_dgrad, false, o->seed);
      dx = torch::mm(g, w_t.t());
    }
    // dW = go^T . x: contraction over tokens
    auto go_t = go.t().contiguous(), in_t = input.t().contiguous();
    torch::Tensor dw;
    if (!o->wgrad)
      dw = mx_matmul(go_t, in_t, torch::kFloat32);
    else if (o->eden_wgrad && eden)
      dw = eden_matmul(go_t, in_t, *o, id, 2);
    else {
      if (o->rht_wgrad)
        go_t = nvfp4_rht(go_t), in_t = nvfp4_rht(in_t);
      dw = at::mm(nvfp4_fake_quant(go_t, o->sr_wgrad, false, o->seed), nvfp4_fake_quant(in_t).t(), torch::kFloat32);
    }
    return {dx, dw.to(weight.scalar_type()), {}, {}, {}};
  }
};

} // namespace

torch::Tensor nvfp4_fake_quant(const torch::Tensor& x_in, bool stochastic, bool blocks_2d, uint64_t seed) {
  torch::NoGradGuard no_grad;
  const auto x = x_in.contiguous();
  TORCH_CHECK(
        x.dim() == 2 && x.is_cuda() && x.size(1) % 16 == 0 &&
              (x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32),
        "nvfp4_fake_quant: expected a 2D CUDA bf16 or fp32 tensor with cols % 16");
  const int64_t R = x.size(0), C = x.size(1);
  const auto amax = x.abs().amax().to(torch::kFloat32);
  torch::Tensor block_amax;
  if (blocks_2d) {
    TORCH_CHECK(R % 16 == 0, "nvfp4_fake_quant: 2D blocks need rows % 16");
    block_amax = x.abs().view({R / 16, 16, C / 16, 16}).amax({1, 3}).to(torch::kFloat32).repeat_interleave(16, 0);
  }
  auto out = torch::empty({R, C}, x.options().dtype(torch::kBFloat16));
  kernels::nvfp4_fake_quant(
        x.data_ptr(), x.scalar_type() == torch::kBFloat16, R, C, amax.data_ptr<float>(),
        blocks_2d ? block_amax.data_ptr<float>() : nullptr, stochastic, seed_counter++ + (seed << 40), out.data_ptr(),
        at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor nvfp4_rht(const torch::Tensor& x) {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 16 == 0, "nvfp4_rht: expected 2D with cols % 16");
  const int64_t R = x.size(0), C = x.size(1);
  return torch::matmul(x.to(torch::kFloat32).reshape({R, C / 16, 16}), nvfp4_hadamard(x.device())).reshape({R, C});
}

Nvfp4Eden nvfp4_eden(const torch::Tensor& x, const torch::Tensor& h, uint64_t seed) {
  torch::NoGradGuard no_grad;
  const int64_t G = h.dim() == 2 ? h.size(0) : 0;
  TORCH_CHECK(
        x.dim() == 2 && G >= 16 && G % 16 == 0 && x.size(1) % G == 0 && h.size(1) == G,
        "nvfp4_eden: expected 2D with cols % G and a G x G rotation, G % 16");
  // Quartet II's backward: block maxima map a little above 6, and tensor-scaled block scales stay <= 256, leaving
  // e4m3 room for the correction
  constexpr float val_max = 6 / (17.f / 16 * 0.93f), scale_max = 255.99f;
  const int64_t R = x.size(0), C = x.size(1);
  const auto xh = torch::matmul(x.to(torch::kFloat32).view({R, C / G, G}), h).view({R, C / 16, 16});
  const auto amax = xh.abs().amax();
  const auto scale = torch::where(amax > 0, amax / (scale_max * val_max), torch::ones_like(amax));
  // block scales, e4m3 round to nearest (0 -> 1)
  auto s = (xh.abs().amax(-1, true) / val_max / scale).to(torch::kFloat8_e4m3fn).to(torch::kFloat32);
  s = torch::where(s > 0, s, torch::ones_like(s));
  const auto xs = xh / (s * scale);
  const auto q = torch::sign(xs) * e2m1_rn(xs.abs());
  // each G-group's ||xs||² / <xs, q> times its block scales
  const auto group_sum = [&](const torch::Tensor& t) {
    return t.view({R, C / G, G}).sum(-1, true);
  };
  const auto num = group_sum(xs * xs), den = group_sum(xs * q);
  const auto corr = torch::where(den > 0, num / den, torch::ones_like(den));
  const auto cs = (s.view({R, C / G, G / 16}) * corr).clamp_max(448).view({R, C / 16, 1});
  // stochastic rounding to one of the two nearest e4m3 values
  const auto bits = cs.to(torch::kFloat8_e4m3fn).view(torch::kUInt8);
  const auto nearest = e4m3_step(bits, 0);
  const auto above = nearest > cs;
  const auto up = torch::where(above, nearest, e4m3_step(bits, 1));
  const auto down = torch::where(above, e4m3_step(bits, -1), nearest);
  const auto p_up = torch::where(up > down, (cs - down) / (up - down), torch::zeros_like(cs));
  auto gen = at::cuda::detail::createCUDAGenerator(x.device().index());
  {
    const std::lock_guard lock(gen.mutex());
    gen.set_current_seed(seed_counter++ + (seed << 40));
  }
  const auto s_sr = torch::where(torch::rand(cs.sizes(), gen, cs.options()) < p_up, up, down);
  return {(q * s_sr).view({R, C}).to(torch::kBFloat16), scale};
}

torch::Tensor nvfp4_eden_rotation(const torch::Device& device, uint64_t seed, int64_t group, bool fresh) {
  TORCH_CHECK(
        group >= 16 && (group & (group - 1)) == 0, "nvfp4_eden_rotation: group must be a power of 2, >= 16: ", group);
  static std::mutex mutex;
  static std::map<std::pair<int, int64_t>, torch::Tensor> cache;
  torch::Tensor h;
  {
    const std::lock_guard lock(mutex);
    auto& c = cache[{device.index(), group}];
    if (!c.defined()) {
      auto m = torch::ones({1, 1});
      for (int64_t n = 1; n < group; n *= 2)
        m = torch::cat({torch::cat({m, m}, 1), torch::cat({m, -m}, 1)}, 0);
      c = (m / std::sqrt(static_cast<double>(group))).to(device);
    }
    h = c;
  }
  // signs before the Hadamard (diag(s) . H): after it they'd cancel in the product, quantization being sign-symmetric
  auto gen = at::make_generator<at::CPUGeneratorImpl>(fresh ? seed_counter++ + (seed << 40) : seed);
  return (torch::randint(0, 2, {group, 1}, gen, torch::kFloat32) * 2 - 1).to(device) * h;
}

torch::Tensor nvfp4_sim_matmul(
      const torch::Tensor& input_2d, const torch::Tensor& weight, const Nvfp4Options& options, int64_t id, bool eden) {
  return Nvfp4SimMatmul::apply(input_2d, weight, &options, id, eden);
}

} // namespace nanochat
