#include "nanochat/model/lambda_blend.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/lambda_blend_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

void check_bf16(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), "expected contiguous bf16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "expected 16-byte aligned data");
}

class LambdaBlend : public torch::autograd::Function<LambdaBlend> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& x0, const torch::Tensor& lr,
        const torch::Tensor& l0, int64_t layer) {
    check_bf16(x);
    check_bf16(x0);
    TORCH_CHECK(x.sizes() == x0.sizes(), "x and x0 must have the same shape");
    TORCH_CHECK(lr.scalar_type() == torch::kFloat32 && l0.scalar_type() == torch::kFloat32);
    TORCH_CHECK(
          layer >= 0 && layer < lr.numel() && lr.numel() == l0.numel() && lr.is_contiguous() && l0.is_contiguous());
    auto out = torch::empty_like(x);
    kernels::lambda_blend_fwd(
          x.data_ptr(), x0.data_ptr(), lr.data_ptr<float>() + layer, l0.data_ptr<float>() + layer, out.data_ptr(),
          x.numel(), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ctx->save_for_backward({x, x0, lr, l0});
    ctx->saved_data["layer"] = layer;
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto saved = ctx->get_saved_variables();
    const auto &x = saved[0], &x0 = saved[1], &lr = saved[2], &l0 = saved[3];
    const auto layer = ctx->saved_data["layer"].toInt();
    const auto g = grads[0].contiguous();
    check_bf16(g);
    auto dx = torch::empty_like(x), dx0 = torch::empty_like(x0);
    auto dlr = torch::empty_like(lr), dl0 = torch::empty_like(l0);
    auto partials = torch::empty({2 * kernels::lambda_blend_bwd_blocks(x.numel())}, lr.options());
    kernels::lambda_blend_bwd(
          g.data_ptr(), x.data_ptr(), x0.data_ptr(), lr.data_ptr<float>() + layer, l0.data_ptr<float>() + layer,
          dx.data_ptr(), dx0.data_ptr(), partials.data_ptr<float>(), dlr.data_ptr<float>(), dl0.data_ptr<float>(),
          static_cast<int>(layer), static_cast<int>(lr.numel()), x.numel(), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dx0, dlr, dl0, torch::Tensor()};
  }
};

} // namespace

torch::Tensor lambda_blend(
      const torch::Tensor& x, const torch::Tensor& x0, const torch::Tensor& resid_lambdas,
      const torch::Tensor& x0_lambdas, int64_t layer) {
  return LambdaBlend::apply(x, x0, resid_lambdas, x0_lambdas, layer);
}

} // namespace nanochat
