#include "nanochat/model/rotary_norm.h"

#include <limits>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/rotary_norm_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

void check_rotary(const torch::Tensor& t, int64_t T, int64_t half) {
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.dim() == 4 && t.size(1) >= T && t.size(3) == half);
  TORCH_CHECK(t.stride(3) == 1 && t.stride(1) == half, "rotary cache must have contiguous rows");
}

class RotaryRmsNorm : public torch::autograd::Function<RotaryRmsNorm> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& x_in, const torch::Tensor& cos, const torch::Tensor& sin,
        double scale) {
    TORCH_CHECK(x_in.scalar_type() == torch::kBFloat16 && x_in.dim() == 4, "x must be (B, T, H, D) bf16");
    const auto x = x_in.contiguous();
    const int64_t T = x.size(1), H = x.size(2), D = x.size(3);
    TORCH_CHECK(D % 4 == 0 && D <= 512, "head_dim must be a multiple of 4, at most 512");
    check_rotary(cos, T, D / 2);
    check_rotary(sin, T, D / 2);
    auto out = torch::empty_like(x);
    auto rstd = torch::empty({x.numel() / D}, x.options().dtype(torch::kFloat32));
    // F.rms_norm's default eps (opmath float epsilon)
    kernels::rotary_norm_fwd(
          x.data_ptr(), cos.data_ptr(), sin.data_ptr(), out.data_ptr(), rstd.data_ptr<float>(), rstd.numel(),
          static_cast<int>(H), T, static_cast<int>(D), static_cast<float>(scale), std::numeric_limits<float>::epsilon(),
          at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ctx->save_for_backward({x, cos, sin, rstd});
    ctx->saved_data["scale"] = scale;
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto saved = ctx->get_saved_variables();
    const auto &x = saved[0], &cos = saved[1], &sin = saved[2], &rstd = saved[3];
    const auto dout = grads[0].contiguous();
    auto dx = torch::empty_like(x);
    kernels::rotary_norm_bwd(
          dout.data_ptr(), x.data_ptr(), cos.data_ptr(), sin.data_ptr(), rstd.data_ptr<float>(), dx.data_ptr(),
          rstd.numel(), static_cast<int>(x.size(2)), x.size(1), static_cast<int>(x.size(3)),
          static_cast<float>(ctx->saved_data["scale"].toDouble()), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

} // namespace

torch::Tensor rotary_rms_norm(
      const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin, double scale) {
  return RotaryRmsNorm::apply(x, cos, sin, scale);
}

} // namespace nanochat
