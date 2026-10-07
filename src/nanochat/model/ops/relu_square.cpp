#include "nanochat/model/ops/relu_square.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/ops/relu_square_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

void check_bf16(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), "expected contiguous bf16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "expected 16-byte aligned data");
}

class ReluSquare : public torch::autograd::Function<ReluSquare> {
public:
  static torch::Tensor forward(AutogradContext* ctx, const torch::Tensor& h) {
    check_bf16(h);
    auto a = torch::empty_like(h);
    kernels::relu_square_fwd(h.data_ptr(), a.data_ptr(), h.numel(), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ctx->save_for_backward({h});
    return a;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto h = ctx->get_saved_variables()[0];
    const auto g = grads[0].contiguous();
    check_bf16(g);
    auto dh = torch::empty_like(h);
    kernels::relu_square_bwd(
          g.data_ptr(), h.data_ptr(), dh.data_ptr(), nullptr, h.numel(), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dh};
  }
};

} // namespace

torch::Tensor relu_square(const torch::Tensor& h) {
  return ReluSquare::apply(h);
}

} // namespace nanochat
