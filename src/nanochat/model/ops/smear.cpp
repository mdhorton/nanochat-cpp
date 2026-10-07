#include "nanochat/model/ops/smear.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/ops/smear_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

class Smear : public torch::autograd::Function<Smear> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& lambda) {
    TORCH_CHECK(
          x.dim() == 3 && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
          "smear: expected a contiguous, aligned (B, T, C) bf16 x");
    TORCH_CHECK(
          w.dim() == 2 && w.size(0) == 1 && w.scalar_type() == torch::kFloat32 && w.is_contiguous() &&
                lambda.numel() == 1 && lambda.scalar_type() == torch::kFloat32,
          "smear: expected an fp32 (1, gate_cols) w and a 1-element fp32 lambda");
    const int64_t T = x.size(1), cols = x.size(2), gate_cols = w.size(1), rows = x.numel() / cols;
    TORCH_CHECK(smear_fits(cols, gate_cols) && gate_cols <= cols, "smear: unsupported dims");
    auto out = torch::empty_like(x);
    auto sig = torch::empty({rows}, x.options().dtype(torch::kFloat32));
    kernels::smear_fwd(
          x.data_ptr(), w.data_ptr<float>(), lambda.data_ptr<float>(), out.data_ptr(), sig.data_ptr<float>(), rows, T,
          static_cast<int>(cols), static_cast<int>(gate_cols), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ctx->save_for_backward({x, w, lambda, sig});
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto saved = ctx->get_saved_variables();
    const auto &x = saved[0], &w = saved[1], &lambda = saved[2], &sig = saved[3];
    const auto g = grads[0].contiguous();
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16 && g.sizes() == x.sizes());
    const int64_t T = x.size(1), cols = x.size(2), gate_cols = w.size(1), rows = sig.numel();
    auto dx = torch::empty_like(x);
    auto dw = torch::empty_like(w), dlambda = torch::empty_like(lambda);
    auto partials = torch::empty({kernels::smear_bwd_blocks(rows) * (gate_cols + 1)}, dw.options());
    kernels::smear_bwd(
          g.data_ptr(), x.data_ptr(), sig.data_ptr<float>(), w.data_ptr<float>(), lambda.data_ptr<float>(),
          dx.data_ptr(), partials.data_ptr<float>(), dw.data_ptr<float>(), dlambda.data_ptr<float>(), rows, T,
          static_cast<int>(cols), static_cast<int>(gate_cols), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dw, dlambda};
  }
};

} // namespace

torch::Tensor smear(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& lambda) {
  return Smear::apply(x, w, lambda);
}

bool smear_fits(int64_t cols, int64_t gate_cols) {
  return cols % 8 == 0 && cols <= kernels::kSmearMaxCols && gate_cols % 8 == 0 && gate_cols > 0 &&
         gate_cols <= kernels::kSmearMaxGateCols;
}

} // namespace nanochat
