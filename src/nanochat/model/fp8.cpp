#include "nanochat/model/fp8.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/fp8_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

constexpr double kEps = 1e-12;

// Returns (fp8 data, inverse scale), as _to_fp8. Python's float / tensor is reciprocal() * float.
std::pair<torch::Tensor, torch::Tensor> to_fp8_reference(const torch::Tensor& x, torch::ScalarType dtype) {
  torch::NoGradGuard no_grad;
  const double fp8_max = dtype == torch::kFloat8_e4m3fn ? 448.0 : 57344.0;
  const auto amax = x.to(torch::kFloat32).abs().max();
  const auto scale = (amax.to(torch::kFloat64).clamp_min(kEps).reciprocal() * fp8_max).to(torch::kFloat32);
  const auto x_fp8 = (x.to(torch::kFloat32) * scale).clamp(-fp8_max, fp8_max).to(dtype);
  return {x_fp8, scale.reciprocal()};
}

// to_fp8_reference in two fused kernels (fp8_kernel.cu), also writing the transpose that backward needs.
Fp8Tensor to_fp8(const torch::Tensor& x_in, torch::ScalarType dtype, bool fused = true) {
  const auto x = x_in.contiguous();
  const bool supported = fused && x.is_cuda() && x.dim() == 2 && x.size(1) % 8 == 0 &&
                         (x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32) &&
                         reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
  if (!supported) {
    auto [data, inv] = to_fp8_reference(x, dtype);
    return {data, data.t().contiguous(), inv};
  }
  const int64_t rows = x.size(0), cols = x.size(1);
  Fp8Tensor q{
        torch::empty({rows, cols}, x.options().dtype(dtype)),
        torch::empty({cols, rows}, x.options().dtype(dtype)),
        {}};
  auto scalars = torch::empty({2}, x.options().dtype(torch::kFloat32)); // amax, inverse scale
  kernels::quantize_fp8(
        x.data_ptr(), x.scalar_type() == torch::kBFloat16, rows, cols,
        dtype == torch::kFloat8_e4m3fn ? kernels::Fp8Format::E4M3 : kernels::Fp8Format::E5M2, q.data.data_ptr(),
        q.data_t.data_ptr(), scalars.data_ptr<float>(), scalars.data_ptr<float>() + 1,
        at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  q.inv_scale = scalars[1];
  return q;
}

// _scaled_mm wants A row-major and B column-major: B = t.t() for a row-major t.
class Float8Matmul : public torch::autograd::Function<Float8Matmul> {
public:
  static torch::Tensor forward(AutogradContext* ctx, const torch::Tensor& input, const torch::Tensor& weight) {
    const auto in = to_fp8(input, torch::kFloat8_e4m3fn);
    const auto w = to_fp8(weight, torch::kFloat8_e4m3fn);
    ctx->save_for_backward({in.data_t, in.inv_scale, w.data_t, w.inv_scale}); // backward's layouts
    // fast accumulation in forward only
    return at::_scaled_mm(in.data, w.data.t(), in.inv_scale, w.inv_scale, {}, {}, input.scalar_type(), true);
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const auto &in_t = saved[0], &in_inv = saved[1], &w_t = saved[2], &w_inv = saved[3];
    const auto& grad_output = grad_outputs[0];
    const auto go = to_fp8(grad_output, torch::kFloat8_e5m2);
    // grad_input = grad_output @ weight, grad_weight = grad_output.T @ input
    auto grad_input = at::_scaled_mm(go.data, w_t.t(), go.inv_scale, w_inv, {}, {}, grad_output.scalar_type(), false);
    auto grad_weight = at::_scaled_mm(
          go.data_t, in_t.t(), go.inv_scale, in_inv, {}, {}, grad_output.scalar_type(), false);
    return {grad_input, grad_weight}; // autograd casts grad_weight to the weight's dtype
  }
};

} // namespace

Fp8Tensor quantize_fp8(const torch::Tensor& x, torch::ScalarType dtype, bool fused) {
  return to_fp8(x, dtype, fused);
}

torch::Tensor fp8_matmul(const torch::Tensor& input_2d, const torch::Tensor& weight) {
  return Float8Matmul::apply(input_2d, weight);
}

} // namespace nanochat
