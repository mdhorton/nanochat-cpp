#include "nanochat/model/fp8.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/fp8_kernel.h"
#include "nanochat/model/qkv_kernel.h"
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

// quantize into given buffers (slices of the merged q/k/v ones); scalars: amax, inverse scale
void quantize_into(const torch::Tensor& x, torch::ScalarType dtype, void* out, void* out_t, float* scalars) {
  TORCH_CHECK(fusable(x) && x.is_contiguous(), "expected an aligned, contiguous 2D tensor");
  kernels::quantize_fp8(
        x.data_ptr(), x.scalar_type() == torch::kBFloat16, x.size(0), x.size(1), format(dtype), out, out_t, scalars,
        scalars + 1, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void* offset(const torch::Tensor& t, int64_t bytes) {
  return static_cast<char*>(t.data_ptr()) + bytes;
}

// Row-wise _scaled_mm scales for merged q/k/v: each part's product of inverse scales (inv[i] * other) over its
// sizes[i] rows. The fp32 product is tensorwise's one scale factor; the other operand's scales are 1, so the GEMM
// rounds as tensorwise (scaling by both vectors rounds twice).
torch::Tensor merged_scales(const torch::Tensor& inv, const torch::Tensor& other, const std::array<int64_t, 3>& sizes) {
  const auto prod = inv * other;
  return torch::cat({prod[0].expand({sizes[0]}), prod[1].expand({sizes[1]}), prod[2].expand({sizes[2]})});
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

std::vector<torch::Tensor> cached(
      Fp8WeightCache* cache, const std::vector<torch::Tensor>& weights,
      const std::function<std::vector<torch::Tensor>()>& make) {
  return cache != nullptr ? cache->get(weights, make) : make();
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
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& input, const torch::Tensor& weight, Fp8WeightCache* cache) {
    const auto in = to_fp8(input, torch::kFloat8_e4m3fn);
    const auto w = quantize_fp8_weight(weight, cache);
    ctx->save_for_backward({in.data_t, in.inv_scale, w.data_t, w.inv_scale}); // backward's layouts
    return mm_forward(in, w, input.scalar_type());
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    const auto s = ctx->get_saved_variables();
    const auto& grad_output = grad_outputs[0];
    const auto go = to_fp8(grad_output, torch::kFloat8_e5m2);
    auto [grad_input, grad_weight] = mm_backward(go, s[0], s[1], s[2], s[3], grad_output.scalar_type());
    return {grad_input, grad_weight, {}}; // autograd casts grad_weight to the weight's dtype
  }
};

// c_proj(relu(c_fc(x)).square()) as two Float8Matmuls, with relu^2 folded into the quantize kernels: forward
// quantizes it straight from h, backward computes dh and its amax in one kernel. Saves h instead of relu(h).
class Fp8ReluSquareMlp : public torch::autograd::Function<Fp8ReluSquareMlp> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& w_fc, const torch::Tensor& w_proj,
        Fp8WeightCache* fc_cache, Fp8WeightCache* proj_cache) {
    const auto xq = to_fp8(x, torch::kFloat8_e4m3fn), fcq = quantize_fp8_weight(w_fc, fc_cache);
    const auto h = mm_forward(xq, fcq, x.scalar_type());
    const auto aq = to_fp8_relu_square(h), projq = quantize_fp8_weight(w_proj, proj_cache);
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
    return {grad_x, grad_fc, grad_proj, {}, {}};
  }
};

// q, k, v = x @ w_i^T as three Float8Matmuls, with x quantized once and one GEMM each for the forward and the weight
// gradients: row-wise scales keep each tensor's own scale, so q, k, v and the weight gradients are bit-identical. The
// input gradient stays three GEMMs (the scales differ along K), summed in one kernel with the gate input's gradient.
class Fp8Qkv : public torch::autograd::Function<Fp8Qkv> {
public:
  static variable_list forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& wq, const torch::Tensor& wk,
        const torch::Tensor& wv, int64_t gate_cols, Fp8WeightCache* cache) {
    const auto e4m3 = torch::kFloat8_e4m3fn;
    const int64_t N = x.size(0), C = x.size(1);
    const std::array<int64_t, 3> sizes{wq.size(0), wk.size(0), wv.size(0)};
    const int64_t n = sizes[0] + sizes[1] + sizes[2];
    const auto xq = to_fp8(x, e4m3);
    // w_cat, w_t[0..2], scalars
    const auto wf = cached(cache, {wq, wk, wv}, [&] {
      const std::array<torch::Tensor, 3> w{wq.contiguous(), wk.contiguous(), wv.contiguous()};
      std::vector<torch::Tensor> out{torch::empty({n, C}, x.options().dtype(e4m3))};
      const auto scalars = torch::empty({3, 2}, x.options().dtype(torch::kFloat32));
      for (int64_t i = 0, row = 0; i < 3; row += sizes[i++]) {
        TORCH_CHECK(w[i].size(1) == C, "q/k/v weights must share the input dim");
        out.push_back(torch::empty({C, sizes[i]}, x.options().dtype(e4m3)));
        quantize_into(w[i], e4m3, offset(out[0], row * C), out.back().data_ptr(), scalars[i].data_ptr<float>());
      }
      out.push_back(scalars);
      return out;
    });
    const auto& w_cat = wf[0];
    const auto w_inv = wf[4].select(1, 1);
    const auto qkv = at::_scaled_mm(
          xq.data, w_cat.t(), torch::ones({N, 1}, w_inv.options()),
          merged_scales(w_inv, xq.inv_scale, sizes).view({1, n}), {}, {}, x.scalar_type(), true);
    ctx->save_for_backward({xq.data_t, xq.inv_scale, wf[1], wf[2], wf[3], w_inv});
    ctx->saved_data["gate_cols"] = gate_cols;
    variable_list out{
          qkv.narrow(1, 0, sizes[0]), qkv.narrow(1, sizes[0], sizes[1]), qkv.narrow(1, sizes[0] + sizes[1], sizes[2])};
    if (gate_cols > 0)
      out.push_back(x.narrow(1, 0, gate_cols).contiguous());
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto e5m2 = torch::kFloat8_e5m2;
    const auto s = ctx->get_saved_variables();
    const auto &x_t = s[0], &x_inv = s[1], &w_inv = s[5];
    const auto gate_cols = ctx->saved_data["gate_cols"].toInt();
    const int64_t C = x_t.size(0), N = x_t.size(1);
    const auto dtype = grads[0].scalar_type();
    std::array<torch::Tensor, 3> g;
    std::array<int64_t, 3> sizes{};
    for (int i = 0; i < 3; ++i) {
      g[i] = grads[i].contiguous();
      sizes[i] = g[i].size(1);
    }
    const int64_t n = sizes[0] + sizes[1] + sizes[2];
    auto g_t = torch::empty({n, N}, g[0].options().dtype(e5m2));
    auto g_scalars = torch::empty({3, 2}, g[0].options().dtype(torch::kFloat32));
    std::array<torch::Tensor, 3> dx;
    for (int64_t i = 0, row = 0; i < 3; row += sizes[i++]) {
      auto gi = torch::empty({N, sizes[i]}, g[i].options().dtype(e5m2));
      quantize_into(g[i], e5m2, gi.data_ptr(), offset(g_t, row * N), g_scalars[i].data_ptr<float>());
      dx[i] = at::_scaled_mm(gi, s[2 + i].t(), g_scalars[i][1], w_inv[i], {}, {}, dtype, false);
    }
    const auto gate = gate_cols > 0 ? grads[3].contiguous() : torch::Tensor();
    auto dx_sum = torch::empty({N, C}, dx[0].options());
    kernels::qkv_grad_sum(
          dx[0].data_ptr(), dx[1].data_ptr(), dx[2].data_ptr(), gate.defined() ? gate.data_ptr() : nullptr,
          static_cast<int>(gate_cols), dx_sum.data_ptr(), N, static_cast<int>(C),
          at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const auto dw = at::_scaled_mm(
          g_t, x_t.t(), merged_scales(g_scalars.select(1, 1), x_inv, sizes).view({n, 1}),
          torch::ones({1, C}, x_inv.options()), {}, {}, dtype, false);
    return {
          dx_sum,
          dw.narrow(0, 0, sizes[0]),
          dw.narrow(0, sizes[0], sizes[1]),
          dw.narrow(0, sizes[0] + sizes[1], sizes[2]),
          torch::Tensor(),
          torch::Tensor()};
  }
};

} // namespace

std::vector<torch::Tensor> Fp8WeightCache::get(
      const std::vector<torch::Tensor>& weights, const std::function<std::vector<torch::Tensor>()>& make) {
  if (!enabled) {
    keys_.clear();
    value_.clear();
    return make();
  }
  bool hit = keys_.size() == weights.size();
  for (size_t i = 0; hit && i < weights.size(); ++i)
    hit = keys_[i].impl == weights[i].unsafeGetTensorImpl() && keys_[i].data == weights[i].data_ptr() &&
          keys_[i].version == weights[i]._version();
  if (!hit) {
    value_ = make();
    keys_.clear();
    for (const auto& w : weights)
      keys_.push_back({w.unsafeGetTensorImpl(), w.data_ptr(), w._version()});
  }
  return value_;
}

Fp8Tensor quantize_fp8(const torch::Tensor& x, torch::ScalarType dtype, bool fused) {
  return to_fp8(x, dtype, fused);
}

Fp8Tensor quantize_fp8_weight(const torch::Tensor& w, Fp8WeightCache* cache) {
  const auto v = cached(cache, {w}, [&] {
    const auto q = to_fp8(w, torch::kFloat8_e4m3fn);
    return std::vector{q.data, q.data_t, q.inv_scale};
  });
  return {v[0], v[1], v[2]};
}

Fp8Tensor quantize_fp8_amax_ready(const torch::Tensor& x, torch::ScalarType dtype, const torch::Tensor& scalars) {
  TORCH_CHECK(fusable(x) && x.is_contiguous(), "expected an aligned, contiguous 2D tensor");
  TORCH_CHECK(scalars.scalar_type() == torch::kFloat32 && scalars.numel() == 2 && scalars.is_contiguous());
  return quantize(x, dtype, scalars, true);
}

torch::Tensor fp8_matmul(const torch::Tensor& input_2d, const torch::Tensor& weight, Fp8WeightCache* cache) {
  return Float8Matmul::apply(input_2d, weight, cache);
}

torch::Tensor fp8_relu_square_mlp(
      const torch::Tensor& x_2d, const torch::Tensor& w_fc, const torch::Tensor& w_proj, Fp8WeightCache* fc_cache,
      Fp8WeightCache* proj_cache) {
  return Fp8ReluSquareMlp::apply(x_2d, w_fc, w_proj, fc_cache, proj_cache);
}

torch::autograd::variable_list fp8_qkv(
      const torch::Tensor& x_2d, const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv,
      int64_t gate_cols, Fp8WeightCache* cache) {
  return Fp8Qkv::apply(x_2d, wq, wk, wv, gate_cols, cache);
}

} // namespace nanochat
