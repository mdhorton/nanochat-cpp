#include "nanochat/model/fp8.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/fp8_kernel.h"
#include "nanochat/model/relu_square_kernel.h"

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

bool fusable(const torch::Tensor& x) {
  return x.is_cuda() && x.dim() == 2 && x.size(1) % 8 == 0 &&
         (x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32) &&
         reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
}

// output buffers for x (rows, cols); scalars: amax, inverse scale
Fp8Tensor empty_fp8(const torch::Tensor& x, torch::ScalarType dtype, const torch::Tensor& scalars) {
  const int64_t rows = x.size(0), cols = x.size(1);
  return {
        torch::empty({rows, cols}, x.options().dtype(dtype)), torch::empty({cols, rows}, x.options().dtype(dtype)),
        scalars[1]};
}

torch::Tensor empty_scalars(const torch::Tensor& x) {
  return torch::empty({2}, x.options().dtype(torch::kFloat32));
}

kernels::Fp8Format format(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e4m3fn ? kernels::Fp8Format::E4M3 : kernels::Fp8Format::E5M2;
}

// amax_ready: scalars[0] already holds max|x|
Fp8Tensor quantize(const torch::Tensor& x, torch::ScalarType dtype, const torch::Tensor& scalars, bool amax_ready) {
  auto q = empty_fp8(x, dtype, scalars);
  kernels::quantize_fp8(
        x.data_ptr(), x.scalar_type() == torch::kBFloat16, x.size(0), x.size(1), format(dtype), q.data.data_ptr(),
        q.data_t.data_ptr(), scalars.data_ptr<float>(), scalars.data_ptr<float>() + 1,
        at::cuda::getCurrentCUDAStream().stream(), amax_ready);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q;
}

// to_fp8_reference in two fused kernels (fp8_kernel.cu), also writing the transpose that backward needs.
Fp8Tensor to_fp8(const torch::Tensor& x_in, torch::ScalarType dtype, bool fused = true) {
  const auto x = x_in.contiguous();
  if (!fused || !fusable(x)) {
    auto [data, inv] = to_fp8_reference(x, dtype);
    return {data, data.t().contiguous(), inv};
  }
  return quantize(x, dtype, empty_scalars(x), false);
}

// to_fp8(relu(h).square(), e4m3) without writing the square
Fp8Tensor to_fp8_relu_square(const torch::Tensor& h) {
  TORCH_CHECK(fusable(h) && h.scalar_type() == torch::kBFloat16, "expected aligned 2D bf16 h");
  const auto scalars = empty_scalars(h);
  auto q = empty_fp8(h, torch::kFloat8_e4m3fn, scalars);
  kernels::quantize_fp8_relu_square(
        h.data_ptr(), h.size(0), h.size(1), q.data.data_ptr(), q.data_t.data_ptr(), scalars.data_ptr<float>(),
        scalars.data_ptr<float>() + 1, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q;
}

// _scaled_mm wants A row-major and B column-major: B = t.t() for a row-major t. Fast accumulation in forward only.
torch::Tensor mm_forward(const Fp8Tensor& in, const Fp8Tensor& w, torch::ScalarType out_dtype) {
  return at::_scaled_mm(in.data, w.data.t(), in.inv_scale, w.inv_scale, {}, {}, out_dtype, true);
}

// grad_input = grad_output @ weight, grad_weight = grad_output.T @ input; in_t, w_t: forward's transposes
std::pair<torch::Tensor, torch::Tensor> mm_backward(
      const Fp8Tensor& go, const torch::Tensor& in_t, const torch::Tensor& in_inv, const torch::Tensor& w_t,
      const torch::Tensor& w_inv, torch::ScalarType out_dtype) {
  return {
        at::_scaled_mm(go.data, w_t.t(), go.inv_scale, w_inv, {}, {}, out_dtype, false),
        at::_scaled_mm(go.data_t, in_t.t(), go.inv_scale, in_inv, {}, {}, out_dtype, false)};
}

class Float8Matmul : public torch::autograd::Function<Float8Matmul> {
public:
  static torch::Tensor forward(AutogradContext* ctx, const torch::Tensor& input, const torch::Tensor& weight) {
    const auto in = to_fp8(input, torch::kFloat8_e4m3fn);
    const auto w = to_fp8(weight, torch::kFloat8_e4m3fn);
    ctx->save_for_backward({in.data_t, in.inv_scale, w.data_t, w.inv_scale}); // backward's layouts
    return mm_forward(in, w, input.scalar_type());
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    const auto s = ctx->get_saved_variables();
    const auto& grad_output = grad_outputs[0];
    const auto go = to_fp8(grad_output, torch::kFloat8_e5m2);
    auto [grad_input, grad_weight] = mm_backward(go, s[0], s[1], s[2], s[3], grad_output.scalar_type());
    return {grad_input, grad_weight}; // autograd casts grad_weight to the weight's dtype
  }
};

// c_proj(relu(c_fc(x)).square()) as two Float8Matmuls, with relu^2 folded into the quantize kernels: forward
// quantizes it straight from h, backward computes dh and its amax in one kernel. Saves h instead of relu(h).
class Fp8ReluSquareMlp : public torch::autograd::Function<Fp8ReluSquareMlp> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& w_fc, const torch::Tensor& w_proj) {
    const auto xq = to_fp8(x, torch::kFloat8_e4m3fn), fcq = to_fp8(w_fc, torch::kFloat8_e4m3fn);
    const auto h = mm_forward(xq, fcq, x.scalar_type());
    const auto aq = to_fp8_relu_square(h), projq = to_fp8(w_proj, torch::kFloat8_e4m3fn);
    ctx->save_for_backward(
          {xq.data_t, xq.inv_scale, fcq.data_t, fcq.inv_scale, h, aq.data_t, aq.inv_scale, projq.data_t,
           projq.inv_scale});
    return mm_forward(aq, projq, x.scalar_type());
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    const auto s = ctx->get_saved_variables();
    const auto& h = s[4];
    const auto& grad_output = grad_outputs[0];
    const auto dtype = grad_output.scalar_type();
    const auto go = to_fp8(grad_output, torch::kFloat8_e5m2);
    const auto [ga, grad_proj] = mm_backward(go, s[5], s[6], s[7], s[8], dtype);
    TORCH_CHECK(ga.scalar_type() == torch::kBFloat16 && ga.is_contiguous());
    auto dh = torch::empty_like(h);
    const auto scalars = empty_scalars(h);
    kernels::relu_square_bwd(
          ga.data_ptr(), h.data_ptr(), dh.data_ptr(), scalars.data_ptr<float>(), h.numel(),
          at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const auto dhq = quantize(dh, torch::kFloat8_e5m2, scalars, true);
    auto [grad_x, grad_fc] = mm_backward(dhq, s[0], s[1], s[2], s[3], dtype);
    return {grad_x, grad_fc, grad_proj};
  }
};

} // namespace

Fp8Tensor quantize_fp8(const torch::Tensor& x, torch::ScalarType dtype, bool fused) {
  return to_fp8(x, dtype, fused);
}

torch::Tensor fp8_matmul(const torch::Tensor& input_2d, const torch::Tensor& weight) {
  return Float8Matmul::apply(input_2d, weight);
}

torch::Tensor fp8_relu_square_mlp(const torch::Tensor& x_2d, const torch::Tensor& w_fc, const torch::Tensor& w_proj) {
  return Fp8ReluSquareMlp::apply(x_2d, w_fc, w_proj);
}

} // namespace nanochat
