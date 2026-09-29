#include "nanochat/model/nvfp4_sim.h"

#include <atomic>

#include <ATen/cuda/CUDAContext.h>

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

class Nvfp4SimMatmul : public torch::autograd::Function<Nvfp4SimMatmul> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& input, const torch::Tensor& weight, const Nvfp4Options* o) {
    ctx->save_for_backward({input, weight});
    ctx->saved_data["o"] = reinterpret_cast<int64_t>(o);
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
    const auto go = grad_outputs[0].contiguous();
    const auto dtype = go.scalar_type();
    // dX = go . W: contraction over out
    torch::Tensor dx;
    if (!o->dgrad)
      dx = mx_matmul(go, weight.t().contiguous(), dtype);
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
    else {
      if (o->rht_wgrad)
        go_t = nvfp4_rht(go_t), in_t = nvfp4_rht(in_t);
      dw = at::mm(nvfp4_fake_quant(go_t, o->sr_wgrad, false, o->seed), nvfp4_fake_quant(in_t).t(), torch::kFloat32);
    }
    return {dx, dw.to(weight.scalar_type()), {}};
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

torch::Tensor nvfp4_sim_matmul(
      const torch::Tensor& input_2d, const torch::Tensor& weight, const Nvfp4Options& options) {
  return Nvfp4SimMatmul::apply(input_2d, weight, &options);
}

} // namespace nanochat
