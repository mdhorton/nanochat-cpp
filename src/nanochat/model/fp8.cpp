#include "nanochat/model/fp8.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/fp8_kernel.h"
#include "nanochat/model/mx_gemm.h"
#include "nanochat/model/nvfp4.h"
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

// kind: what make() makes (the recipe, and for Mx whether data_t is NVFP4)
std::vector<torch::Tensor> cached(
      Fp8WeightCache* cache, int64_t kind, const std::vector<torch::Tensor>& weights,
      const std::function<std::vector<torch::Tensor>()>& make) {
  return cache != nullptr ? cache->get(kind, weights, make) : make();
}

// whether Mx weights' transposes are NVFP4 (for NVFP4 dgrad)
bool fp4_weights_t() {
  const auto* o = nvfp4_backward();
  return o != nullptr && o->dgrad;
}

int64_t mx_kind(bool fp4_t, bool fp4) {
  return static_cast<int64_t>(Fp8Recipe::Mx) + (fp4_t ? 2 : 0) + (fp4 ? 4 : 0);
}

// Weights (rows concatenated: q's (n, C), from empty_mx) quantized into q, finished. fp4: data NVFP4 in 16x16 blocks,
// data_t too when NVFP4 (the same blocks), else MX.
void quantize_weights_into(Fp8Tensor& q, const std::vector<torch::Tensor>& ws, bool fp4) {
  int64_t row = 0;
  for (const auto& w_in : ws) {
    const auto w = w_in.contiguous();
    kernels::MxOut out{}, out_t{};
    if (q.fp4_target.data.defined())
      out.fp4 = nvfp4_out(q.fp4_target, row, 0);
    else
      out = mx_out(q.data, q.inv_scale, row, 0);
    if (q.fp4_target_t.data.defined())
      out_t.fp4 = nvfp4_out(q.fp4_target_t, 0, row);
    else
      out_t = mx_out(q.data_t, q.inv_scale_t, 0, row);
    if (fp4) {
      quantize_nvfp4_2d(w, out.fp4, out_t.fp4);
      if (out_t.fp4.data == nullptr)
        quantize_mx_into(w, false, {}, out_t);
    }
    else
      quantize_mx_into(w, false, out, out_t);
    row += w.size(0);
  }
  finish_fp4(q);
}

} // namespace

torch::Tensor empty_mx_scale(int64_t rows, int64_t cols, const torch::TensorOptions& options) {
  return torch::empty({rows * cols / 32}, options.dtype(torch::kFloat8_e8m0fnu));
}

kernels::MxOut mx_out(const torch::Tensor& data, const torch::Tensor& scale, int64_t row, int64_t col) {
  const int64_t ld = data.size(1), tiles = ld / 128;
  return {offset(data, row * ld + col), ld, offset(scale, ((row / 128) * tiles + col / 128) * 512), tiles};
}

Fp8Tensor empty_mx(
      int64_t R, int64_t C, const torch::TensorOptions& options, bool rows, bool cols, Nvfp4Role role,
      Nvfp4Role role_t) {
  Fp8Tensor q;
  if (rows)
    q.fp4_target = nvfp4_target(R, C, options, role);
  if (rows && !q.fp4_target.data.defined()) {
    q.data = torch::empty({R, C}, options.dtype(torch::kFloat8_e4m3fn));
    q.inv_scale = empty_mx_scale(R, C, options);
  }
  if (cols)
    q.fp4_target_t = nvfp4_target(C, R, options, role_t);
  if (cols && !q.fp4_target_t.data.defined()) {
    q.data_t = torch::empty({C, R}, options.dtype(torch::kFloat8_e4m3fn));
    q.inv_scale_t = empty_mx_scale(C, R, options);
  }
  return q;
}

std::pair<kernels::MxOut, kernels::MxOut> mx_outs(const Fp8Tensor& q) {
  kernels::MxOut out{}, out_t{};
  if (q.fp4_target.data.defined())
    out.fp4 = nvfp4_out(q.fp4_target);
  else if (q.data.defined())
    out = mx_out(q.data, q.inv_scale, 0, 0);
  if (q.fp4_target_t.data.defined())
    out_t.fp4 = nvfp4_out(q.fp4_target_t);
  else if (q.data_t.defined())
    out_t = mx_out(q.data_t, q.inv_scale_t, 0, 0);
  return {out, out_t};
}

void finish_fp4(Fp8Tensor& q) {
  if (q.fp4_target.data.defined() && q.fp4_target_t.data.defined()) { // one launch
    const auto [t, t_t] = nvfp4_finish(q.fp4_target, q.fp4_target_t);
    q.data = t.data, q.inv_scale = t.scale, q.amax = t.amax;
    q.data_t = t_t.data, q.inv_scale_t = t_t.scale, q.amax_t = t_t.amax;
    q.fp4_target = {}, q.fp4_target_t = {};
    return;
  }
  if (q.fp4_target.data.defined()) {
    const auto t = nvfp4_finish(q.fp4_target);
    q.data = t.data, q.inv_scale = t.scale, q.amax = t.amax;
    q.fp4_target = {};
  }
  if (q.fp4_target_t.data.defined()) {
    const auto t = nvfp4_finish(q.fp4_target_t);
    q.data_t = t.data, q.inv_scale_t = t.scale, q.amax_t = t.amax;
    q.fp4_target_t = {};
  }
}

void quantize_mx_into(const torch::Tensor& x, bool relu_square, kernels::MxOut out, kernels::MxOut out_t) {
  TORCH_CHECK(
        fusable(x) && x.is_contiguous() && mx_fits(x.size(0), x.size(1)) &&
              (!relu_square || x.scalar_type() == torch::kBFloat16),
        "MX: expected an aligned, contiguous 2D tensor with dims % 128");
  kernels::quantize_mx(
        x.data_ptr(), x.scalar_type() == torch::kBFloat16, x.size(0), x.size(1), out, out_t, relu_square,
        at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> mx_qkv_weights(
      const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv, Fp8WeightCache* cache, bool fp4) {
  const int64_t C = wq.size(1), n = wq.size(0) + wk.size(0) + wv.size(0);
  for (const auto& w : {wk, wv})
    TORCH_CHECK(w.size(1) == C, "q/k/v weights must share the input dim");
  return cached(cache, mx_kind(fp4_weights_t(), fp4), {wq, wk, wv}, [&] {
    auto q = empty_mx(
          n, C, wq.options(), true, true, fp4 ? Nvfp4Role::FwdInput : Nvfp4Role::None, Nvfp4Role::DgradWeight);
    quantize_weights_into(q, {wq, wk, wv}, fp4);
    return std::vector{q.data, q.data_t, q.inv_scale, q.inv_scale_t, q.amax, q.amax_t};
  });
}

namespace {

bool is_mx(const Fp8Tensor& t) {
  return t.inv_scale.scalar_type() == torch::kFloat8_e8m0fnu || t.fp4();
}

// fp4_t: the transpose as NVFP4 when NVFP4 weight gradients are on (MX, the weight's gradient written directly). fp4:
// the rows as NVFP4 (NVFP4 forward).
Fp8Tensor quantize_input(const torch::Tensor& x, bool mx, bool fp4_t, bool fp4 = false) {
  return mx ? quantize_mx(
                    x, true, true, false, fp4 ? Nvfp4Role::FwdInput : Nvfp4Role::None,
                    fp4_t ? Nvfp4Role::WgradInput : Nvfp4Role::None)
            : to_fp8(x, torch::kFloat8_e4m3fn);
}

// tensorwise: e5m2; MX: e4m3 (block scales cover the range), NVFP4 for NVFP4 dgrad; the transpose as quantize_input's
Fp8Tensor quantize_grad(const torch::Tensor& g, bool mx, bool fp4) {
  return mx ? quantize_mx(g, true, true, false, Nvfp4Role::DgradGrad, fp4 ? Nvfp4Role::WgradGrad : Nvfp4Role::None)
            : to_fp8(g, torch::kFloat8_e5m2);
}

Fp8Recipe recipe_of(bool mx) {
  return mx ? Fp8Recipe::Mx : Fp8Recipe::Tensorwise;
}

// rows [row, row + size) of dw, or undefined when the gradient went into .grad
torch::Tensor part(const torch::Tensor& dw, int64_t row, int64_t size) {
  return dw.defined() ? dw.narrow(0, row, size) : torch::Tensor();
}

// _scaled_mm wants A row-major and B column-major: B = t.t() for a row-major t. Fast accumulation in forward only
// (tensorwise).
torch::Tensor mm_forward(const Fp8Tensor& in, const Fp8Tensor& w, torch::ScalarType out_dtype) {
  TORCH_CHECK(in.fp4() == w.fp4(), "NVFP4 forward: both operands or neither");
  if (in.fp4()) {
    TORCH_CHECK(out_dtype == torch::kBFloat16, "NVFP4 forward: bf16 outputs only");
    return nvfp4_gemm(in.nvfp4(), w.nvfp4());
  }
  if (is_mx(in))
    return mx_gemm(in.data, in.inv_scale, w.data, w.inv_scale, out_dtype);
  return at::_scaled_mm(in.data, w.data.t(), in.inv_scale, w.inv_scale, {}, {}, out_dtype, true);
}

// grad_input = grad_output @ weight; w_t: forward's transpose (w_amax: NVFP4's)
torch::Tensor mm_grad_input(
      const Fp8Tensor& go, const torch::Tensor& w_t, const torch::Tensor& w_inv, const torch::Tensor& w_amax,
      torch::ScalarType out_dtype) {
  const bool fp4 = w_t.scalar_type() == torch::kUInt8;
  TORCH_CHECK(go.fp4() == fp4, "NVFP4 dgrad: both operands or neither");
  if (fp4) {
    TORCH_CHECK(out_dtype == torch::kBFloat16, "NVFP4 dgrad: bf16 gradients only");
    return nvfp4_gemm(go.nvfp4(), {w_t, w_inv, w_amax});
  }
  if (is_mx(go))
    return mx_gemm(go.data, go.inv_scale, w_t, w_inv, out_dtype);
  return at::_scaled_mm(go.data, w_t.t(), go.inv_scale, w_inv, {}, {}, out_dtype, false);
}

// grad_weight = grad_output.T @ input; in_t: forward's transpose (in_amax: NVFP4's). With params (MX: the weights
// themselves, saved by forward), written straight into their .grad and undefined is returned; else the gradient for
// autograd (which casts it to the weight's dtype).
torch::Tensor mm_grad_weight(
      const Fp8Tensor& go, const torch::Tensor& in_t, const torch::Tensor& in_inv, const torch::Tensor& in_amax,
      torch::ScalarType out_dtype, const std::vector<torch::Tensor>& params) {
  if (params.empty()) {
    TORCH_CHECK(!go.fp4_t() && !in_amax.defined(), "NVFP4 weight gradients need the weights (mx_grad_direct)");
    return is_mx(go) ? mx_gemm(go.data_t, go.inv_t(), in_t, in_inv, out_dtype)
                     : at::_scaled_mm(go.data_t, in_t.t(), go.inv_t(), in_inv, {}, {}, out_dtype, false);
  }
  mx_grad_weights(go.data_t, go.inv_t(), in_t, in_inv, params, {}, true, go.amax_t, in_amax);
  return {};
}

// forward saves the weights whose .grad backward writes itself (MX; all of ws or none)
void save_grad_params(AutogradContext* ctx, const char* key, bool mx, const std::vector<torch::Tensor>& ws) {
  if (!mx)
    return;
  for (const auto& w : ws)
    if (!mx_grad_direct(w))
      return;
  ctx->saved_data[key] = ws;
}

std::vector<torch::Tensor> grad_params(AutogradContext* ctx, const char* key) {
  return ctx->saved_data.count(key) != 0 ? ctx->saved_data[key].toTensorVector() : std::vector<torch::Tensor>{};
}

class Float8Matmul : public torch::autograd::Function<Float8Matmul> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& input, const torch::Tensor& weight, Fp8WeightCache* cache, bool mx,
        bool fp4) {
    const auto in = quantize_input(input, mx, mx && mx_grad_direct(weight), fp4);
    const auto w = quantize_fp8_weight(weight, cache, recipe_of(mx), true, fp4);
    ctx->save_for_backward({in.data_t, in.inv_t(), w.data_t, w.inv_t(), in.amax_t, w.amax_t}); // backward's layouts
    ctx->saved_data["mx"] = mx;
    save_grad_params(ctx, "weight", mx, {weight});
    return mm_forward(in, w, input.scalar_type());
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    const auto s = ctx->get_saved_variables();
    const auto& grad_output = grad_outputs[0];
    const auto dtype = grad_output.scalar_type();
    const auto params = grad_params(ctx, "weight");
    const auto go = quantize_grad(grad_output, ctx->saved_data["mx"].toBool(), !params.empty());
    const auto grad_input = mm_grad_input(go, s[2], s[3], s[5], dtype);
    return {grad_input, mm_grad_weight(go, s[0], s[1], s[4], dtype, params), {}, {}, {}};
  }
};

// c_proj(relu(c_fc(x)).square()) as two Float8Matmuls, with relu^2 folded into the quantize kernels: forward
// quantizes it straight from h (MX + CUTLASS: in c_fc's epilogue, NVFP4 with fp4), backward computes dh with its amax
// (tensorwise) or quantizes it without writing it (MX). Saves h instead of relu(h).
class Fp8ReluSquareMlp : public torch::autograd::Function<Fp8ReluSquareMlp> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& w_fc, const torch::Tensor& w_proj,
        Fp8WeightCache* fc_cache, Fp8WeightCache* proj_cache, bool mx, const Fp8Tensor* x_mx, bool fp4) {
    TORCH_CHECK(x_mx == nullptr || (mx && x_mx->data.defined() && x_mx->data_t.defined()), "x_mx needs Mx");
    TORCH_CHECK(x_mx == nullptr || x_mx->fp4() == fp4, "x_mx: NVFP4 rows with fp4");
    const auto xq = x_mx != nullptr ? *x_mx : quantize_input(x, mx, mx && mx_grad_direct(w_fc), fp4);
    const auto fcq = quantize_fp8_weight(w_fc, fc_cache, recipe_of(mx), true, fp4);
    torch::Tensor h;
    Fp8Tensor aq;
    if (mx) {
      // CUTLASS: the GEMM's epilogue quantizes too
      h = torch::empty({x.size(0), w_fc.size(0)}, x.options());
      aq = empty_mx(
            h.size(0), h.size(1), h.options(), true, true, fp4 ? Nvfp4Role::FwdInput : Nvfp4Role::None,
            mx_grad_direct(w_proj) ? Nvfp4Role::WgradInput : Nvfp4Role::None);
      const auto [out, out_t] = mx_outs(aq);
      if (fp4 ? !nvfp4_gemm_relu_square(xq.nvfp4(), fcq.nvfp4(), h, out, out_t)
              : !mx_gemm_relu_square(
                      h, aq.data, aq.inv_scale, aq.data_t, aq.inv_scale_t, xq.data, xq.inv_scale, fcq.data,
                      fcq.inv_scale, out_t.fp4)) {
        if (fp4)
          h = nvfp4_gemm(xq.nvfp4(), fcq.nvfp4());
        else
          mx_gemm_out(h, xq.data, xq.inv_scale, fcq.data, fcq.inv_scale);
        quantize_mx_into(h, true, out, out_t);
      }
      finish_fp4(aq);
    }
    else {
      h = mm_forward(xq, fcq, x.scalar_type());
      aq = to_fp8_relu_square(h);
    }
    const auto projq = quantize_fp8_weight(w_proj, proj_cache, recipe_of(mx), true, fp4);
    ctx->save_for_backward(
          {xq.data_t, xq.inv_t(), fcq.data_t, fcq.inv_t(), h, aq.data_t, aq.inv_t(), projq.data_t, projq.inv_t(),
           xq.amax_t, aq.amax_t, fcq.amax_t, projq.amax_t});
    ctx->saved_data["mx"] = mx;
    save_grad_params(ctx, "w_fc", mx, {w_fc});
    save_grad_params(ctx, "w_proj", mx, {w_proj});
    return mm_forward(aq, projq, x.scalar_type());
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    const auto s = ctx->get_saved_variables();
    const bool mx = ctx->saved_data["mx"].toBool();
    const auto& h = s[4];
    const auto& grad_output = grad_outputs[0];
    const auto dtype = grad_output.scalar_type();
    const auto fc_params = grad_params(ctx, "w_fc"), proj_params = grad_params(ctx, "w_proj");
    const auto go = quantize_grad(grad_output, mx, !proj_params.empty());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto compute_ga = [&] {
      const auto ga = mm_grad_input(go, s[7], s[8], s[12], dtype);
      TORCH_CHECK(ga.scalar_type() == torch::kBFloat16 && ga.is_contiguous() && h.is_contiguous());
      return ga;
    };
    const auto grad_proj = mm_grad_weight(go, s[5], s[6], s[10], dtype, proj_params);
    Fp8Tensor dhq;
    if (mx) {
      dhq = empty_mx(
            h.size(0), h.size(1), h.options(), true, true, go.fp4() ? Nvfp4Role::DgradGrad : Nvfp4Role::None,
            fc_params.empty() ? Nvfp4Role::None : Nvfp4Role::WgradGrad);
      const auto [out, out_t] = mx_outs(dhq);
      // CUTLASS: the dgrad GEMM's epilogue does it all
      const bool fused = dtype == torch::kBFloat16 &&
                         (go.fp4() ? nvfp4_gemm_relu_square_bwd(go.nvfp4(), {s[7], s[8], s[12]}, h, out, out_t)
                                   : mx_gemm_relu_square_bwd(
                                           go.data, go.inv_scale, s[7], s[8], h, dhq.data, dhq.inv_scale, dhq.data_t,
                                           dhq.inv_scale_t, out_t.fp4));
      if (!fused) {
        const auto ga = compute_ga();
        kernels::quantize_mx_relu_square_bwd(ga.data_ptr(), h.data_ptr(), h.size(0), h.size(1), out, out_t, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      finish_fp4(dhq);
    }
    else {
      const auto ga = compute_ga();
      auto dh = torch::empty_like(h);
      const auto scalars = empty_scalars(h);
      kernels::relu_square_bwd(
            ga.data_ptr(), h.data_ptr(), dh.data_ptr(), scalars.data_ptr<float>(), h.numel(), stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      dhq = quantize(dh, torch::kFloat8_e5m2, scalars, true);
    }
    const auto grad_x = mm_grad_input(dhq, s[2], s[3], s[11], dtype);
    return {grad_x, mm_grad_weight(dhq, s[0], s[1], s[9], dtype, fc_params), grad_proj, {}, {}, {}, {}, {}};
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
    const auto wf = cached(cache, static_cast<int64_t>(Fp8Recipe::Tensorwise), {wq, wk, wv}, [&] {
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

// Fp8Qkv with MX scales, which are per block along K: the parts concatenate along either dim, so the input gradient is
// one GEMM too (plus the gate input's gradient).
class MxQkv : public torch::autograd::Function<MxQkv> {
public:
  static variable_list forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& wq, const torch::Tensor& wk,
        const torch::Tensor& wv, int64_t gate_cols, Fp8WeightCache* cache) {
    const std::array<int64_t, 3> sizes{wq.size(0), wk.size(0), wv.size(0)};
    const auto xq = quantize_mx(x);
    const auto wf = mx_qkv_weights(wq, wk, wv, cache);
    const auto qkv = mx_gemm(xq.data, xq.inv_scale, wf[0], wf[2], x.scalar_type());
    ctx->save_for_backward({xq.data_t, xq.inv_scale_t, wf[1], wf[3], wf[5]});
    ctx->saved_data["gate_cols"] = gate_cols;
    save_grad_params(ctx, "weights", true, {wq, wk, wv});
    variable_list out{
          qkv.narrow(1, 0, sizes[0]), qkv.narrow(1, sizes[0], sizes[1]), qkv.narrow(1, sizes[0] + sizes[1], sizes[2])};
    if (gate_cols > 0)
      out.push_back(x.narrow(1, 0, gate_cols).contiguous());
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto s = ctx->get_saved_variables();
    const auto &x_t = s[0], &x_scale_t = s[1], &w_t = s[2], &w_scale_t = s[3], &w_amax = s[4];
    const auto gate_cols = ctx->saved_data["gate_cols"].toInt();
    const int64_t N = x_t.size(1);
    const auto dtype = grads[0].scalar_type();
    std::array<int64_t, 3> sizes{grads[0].size(1), grads[1].size(1), grads[2].size(1)};
    const int64_t n = sizes[0] + sizes[1] + sizes[2];
    // the gradient (N, n): NVFP4 when the weights' transpose is (NVFP4 dgrad)
    auto g = empty_mx(
          N, n, grads[0].options(), true, true,
          w_t.scalar_type() == torch::kUInt8 ? Nvfp4Role::DgradGrad : Nvfp4Role::None);
    for (int64_t i = 0, row = 0; i < 3; row += sizes[i++]) {
      kernels::MxOut out{};
      if (g.fp4_target.data.defined())
        out.fp4 = nvfp4_out(g.fp4_target, 0, row);
      else
        out = mx_out(g.data, g.inv_scale, 0, row);
      quantize_mx_into(grads[i].contiguous(), false, out, mx_out(g.data_t, g.inv_scale_t, row, 0));
    }
    finish_fp4(g);
    auto dx = mm_grad_input(g, w_t, w_scale_t, w_amax, dtype);
    if (gate_cols > 0)
      dx.narrow(1, 0, gate_cols).add_(grads[3]);
    const auto dw = mm_grad_weight(g, x_t, x_scale_t, {}, dtype, grad_params(ctx, "weights"));
    return {dx, part(dw, 0, sizes[0]), part(dw, sizes[0], sizes[1]), part(dw, sizes[0] + sizes[1], sizes[2]), {}, {}};
  }
};

} // namespace

bool mx_grad_direct(const torch::Tensor& w) {
  return w.is_leaf() && w.requires_grad() && w.scalar_type() == torch::kFloat32 && w.is_contiguous();
}

void mx_grad_weights(
      const torch::Tensor& go_t, const torch::Tensor& go_scale_t, const torch::Tensor& in_t,
      const torch::Tensor& in_scale_t, const std::vector<torch::Tensor>& ws, const torch::Tensor& alpha, bool nvfp4,
      const torch::Tensor& go_amax, const torch::Tensor& in_amax) {
  const bool fp4 = go_amax.defined() || in_amax.defined() ||
                   (nvfp4 && nvfp4_backward() != nullptr && nvfp4_backward()->wgrad);
  const auto gemm = [&](const torch::Tensor& out, bool accumulate) {
    if (fp4)
      nvfp4_grad_weight(go_t, go_scale_t, go_amax, in_t, in_scale_t, in_amax, out, accumulate, alpha);
    else
      mx_gemm_f32(go_t, go_scale_t, in_t, in_scale_t, out, accumulate, alpha);
  };
  const int64_t C = in_t.size(0);
  int64_t n = 0;
  size_t defined = 0;
  for (const auto& w : ws) {
    TORCH_CHECK(w.dim() == 2 && w.size(1) == C && w.scalar_type() == torch::kFloat32);
    n += w.size(0);
    defined += w.grad().defined();
  }
  TORCH_CHECK(go_t.size(0) == n, "mx_grad_weights: the weights' rows must add up to the gradient's");
  // the .grads as the row blocks of one buffer: fresh, or the ones a previous call set up
  torch::Tensor buf;
  if (defined == 0) {
    buf = torch::empty({n, C}, ws[0].options());
    for (int64_t i = 0, row = 0; i < static_cast<int64_t>(ws.size()); row += ws[i++].size(0))
      ws[i].mutable_grad() = ws.size() == 1 ? buf : buf.narrow(0, row, ws[i].size(0));
  }
  else if (defined == ws.size()) {
    const auto& g0 = ws[0].grad();
    bool blocks = g0.is_contiguous() && g0.scalar_type() == torch::kFloat32;
    for (int64_t i = 1, row = ws[0].size(0); blocks && i < static_cast<int64_t>(ws.size()); row += ws[i++].size(0)) {
      const auto& g = ws[i].grad();
      blocks = g.is_contiguous() && g.scalar_type() == torch::kFloat32 && g.is_alias_of(g0) &&
               g.data_ptr() == static_cast<char*>(g0.data_ptr()) + row * C * static_cast<int64_t>(sizeof(float));
    }
    if (blocks)
      buf = g0.as_strided({n, C}, {C, 1});
  }
  if (buf.defined()) {
    gemm(buf, defined > 0);
    return;
  }
  // .grads set up elsewhere: one GEMM, then adds
  const auto dw = torch::empty({n, C}, ws[0].options());
  gemm(dw, false);
  for (int64_t i = 0, row = 0; i < static_cast<int64_t>(ws.size()); row += ws[i++].size(0)) {
    const auto p = dw.narrow(0, row, ws[i].size(0));
    if (ws[i].grad().defined())
      ws[i].mutable_grad().add_(p);
    else
      ws[i].mutable_grad() = p;
  }
}

std::vector<torch::Tensor> Fp8WeightCache::get(
      int64_t kind, const std::vector<torch::Tensor>& weights,
      const std::function<std::vector<torch::Tensor>()>& make) {
  if (!enabled) {
    keys_.clear();
    value_.clear();
    return make();
  }
  bool hit = kind_ == kind && keys_.size() == weights.size();
  for (size_t i = 0; hit && i < weights.size(); ++i)
    hit = keys_[i].impl == weights[i].unsafeGetTensorImpl() && keys_[i].data == weights[i].data_ptr() &&
          keys_[i].version == weights[i]._version();
  if (!hit) {
    value_ = make();
    kind_ = kind;
    keys_.clear();
    for (const auto& w : weights)
      keys_.push_back({w.unsafeGetTensorImpl(), w.data_ptr(), w._version()});
  }
  return value_;
}

Fp8Tensor quantize_fp8(const torch::Tensor& x, torch::ScalarType dtype, bool fused) {
  return to_fp8(x, dtype, fused);
}

Fp8Tensor quantize_mx(
      const torch::Tensor& x_in, bool rows, bool cols, bool relu_square, Nvfp4Role role, Nvfp4Role role_t) {
  const auto x = x_in.contiguous();
  TORCH_CHECK(x.dim() == 2, "MX: expected a 2D tensor");
  auto q = empty_mx(x.size(0), x.size(1), x.options(), rows, cols, role, role_t);
  const auto [out, out_t] = mx_outs(q);
  quantize_mx_into(x, relu_square, out, out_t);
  finish_fp4(q);
  return q;
}

Fp8Tensor quantize_fp8_weight(const torch::Tensor& w, Fp8WeightCache* cache, Fp8Recipe recipe, bool fp4_t, bool fp4) {
  const bool mx = recipe == Fp8Recipe::Mx;
  fp4 = mx && fp4;
  fp4_t = mx && fp4_t && fp4_weights_t();
  const auto v = cached(cache, mx ? mx_kind(fp4_t, fp4) : static_cast<int64_t>(recipe), {w}, [&] {
    Fp8Tensor q;
    if (mx) {
      TORCH_CHECK(w.dim() == 2, "MX: expected a 2D weight");
      q = empty_mx(
            w.size(0), w.size(1), w.options(), true, true, fp4 ? Nvfp4Role::FwdInput : Nvfp4Role::None,
            fp4_t ? Nvfp4Role::DgradWeight : Nvfp4Role::None);
      quantize_weights_into(q, {w}, fp4);
    }
    else
      q = to_fp8(w, torch::kFloat8_e4m3fn);
    return std::vector{q.data, q.data_t, q.inv_scale, q.inv_scale_t, q.amax, q.amax_t};
  });
  return {.data = v[0], .data_t = v[1], .inv_scale = v[2], .inv_scale_t = v[3], .amax = v[4], .amax_t = v[5]};
}

Fp8Tensor quantize_fp8_amax_ready(const torch::Tensor& x, torch::ScalarType dtype, const torch::Tensor& scalars) {
  TORCH_CHECK(fusable(x) && x.is_contiguous(), "expected an aligned, contiguous 2D tensor");
  TORCH_CHECK(scalars.scalar_type() == torch::kFloat32 && scalars.numel() == 2 && scalars.is_contiguous());
  return quantize(x, dtype, scalars, true);
}

torch::Tensor fp8_matmul(
      const torch::Tensor& input_2d, const torch::Tensor& weight, Fp8WeightCache* cache, Fp8Recipe recipe, bool fp4) {
  const bool mx = recipe == Fp8Recipe::Mx && mx_fits(input_2d.size(0), input_2d.size(1)) &&
                  mx_fits(weight.size(0), weight.size(1));
  return Float8Matmul::apply(
        input_2d, weight, cache, mx,
        fp4 && mx && nvfp4_fits(input_2d.size(1)) && input_2d.scalar_type() == torch::kBFloat16);
}

bool relu_square_mlp_mx(
      int64_t N, int64_t in, const torch::Tensor& w_fc, const torch::Tensor& w_proj, Fp8Recipe recipe) {
  return recipe == Fp8Recipe::Mx && mx_fits(N, in) && mx_fits(w_fc.size(0), w_fc.size(1)) &&
         mx_fits(w_proj.size(0), w_proj.size(1));
}

torch::Tensor fp8_relu_square_mlp(
      const torch::Tensor& x_2d, const torch::Tensor& w_fc, const torch::Tensor& w_proj, Fp8WeightCache* fc_cache,
      Fp8WeightCache* proj_cache, Fp8Recipe recipe, const Fp8Tensor* x_mx, bool fp4) {
  const bool mx = relu_square_mlp_mx(x_2d.size(0), x_2d.size(1), w_fc, w_proj, recipe);
  TORCH_CHECK(
        !fp4 || (mx && nvfp4_fits(x_2d.size(1)) && nvfp4_fits(w_fc.size(0)) && x_2d.scalar_type() == torch::kBFloat16),
        "fp8_relu_square_mlp: NVFP4 needs Mx, bf16 and in, hidden % 256");
  return Fp8ReluSquareMlp::apply(x_2d, w_fc, w_proj, fc_cache, proj_cache, mx, x_mx, fp4);
}

torch::autograd::variable_list fp8_qkv(
      const torch::Tensor& x_2d, const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv,
      int64_t gate_cols, Fp8WeightCache* cache, Fp8Recipe recipe) {
  const auto fits = [](const torch::Tensor& t) {
    return mx_fits(t.size(0), t.size(1));
  };
  if (recipe == Fp8Recipe::Mx && fits(x_2d) && fits(wq) && fits(wk) && fits(wv))
    return MxQkv::apply(x_2d, wq, wk, wv, gate_cols, cache);
  return Fp8Qkv::apply(x_2d, wq, wk, wv, gate_cols, cache);
}

} // namespace nanochat
