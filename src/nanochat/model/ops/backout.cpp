#include "nanochat/model/ops/backout.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/ops/backout_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

void check_bf16(const torch::Tensor& t, const torch::Tensor& like) {
  TORCH_CHECK(
        t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.sizes() == like.sizes() &&
              reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
        "backout: expected same-shape, contiguous, aligned bf16 tensors");
}

class Backout : public torch::autograd::Function<Backout> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& xb, const torch::Tensor& lambda) {
    check_bf16(x, x);
    check_bf16(xb, x);
    TORCH_CHECK(
          x.numel() % 8 == 0 && lambda.numel() == 1 && lambda.scalar_type() == torch::kFloat32,
          "backout: numel % 8 == 0, fp32 lambda");
    auto out = torch::empty_like(x);
    kernels::backout_fwd(
          x.data_ptr(), xb.data_ptr(), lambda.data_ptr<float>(), out.data_ptr(), x.numel(),
          at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ctx->save_for_backward({xb, lambda});
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto saved = ctx->get_saved_variables();
    const auto &xb = saved[0], &lambda = saved[1];
    const auto g = grads[0].contiguous();
    check_bf16(g, xb);
    auto dxb = torch::empty_like(xb);
    auto dlambda = torch::empty_like(lambda);
    auto partials = torch::empty({kernels::backout_bwd_blocks(xb.numel())}, dlambda.options());
    kernels::backout_bwd(
          g.data_ptr(), xb.data_ptr(), lambda.data_ptr<float>(), dxb.data_ptr(), partials.data_ptr<float>(),
          dlambda.data_ptr<float>(), xb.numel(), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g, dxb, dlambda};
  }
};

} // namespace

torch::Tensor backout(const torch::Tensor& x, const torch::Tensor& xb, const torch::Tensor& lambda) {
  return Backout::apply(x, xb, lambda);
}

} // namespace nanochat
