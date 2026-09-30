#include "nanochat/model/residual_norm.h"

#include <limits>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/residual_norm_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

void check_bf16(const torch::Tensor& t, const torch::Tensor& like) {
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), "expected contiguous bf16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "expected 16-byte aligned data");
  TORCH_CHECK(t.sizes() == like.sizes(), "shape mismatch");
}

const void* ptr(const torch::Tensor& t) {
  return t.defined() ? t.data_ptr() : nullptr;
}

// autograd::Function inputs can't be undefined tensors, but can be empty optionals
using Optional = std::optional<torch::Tensor>;

Optional optional(const torch::Tensor& t) {
  return t.defined() ? Optional(t) : std::nullopt;
}

class ResidualNorm : public torch::autograd::Function<ResidualNorm> {
public:
  static variable_list forward(
        AutogradContext* ctx, const torch::Tensor& x, const Optional& r_in, const Optional& x0_in,
        const Optional& lr_in, const Optional& l0_in, int64_t layer, const c10::intrusive_ptr<X0Grad>& x0_grad, bool mx,
        int64_t gate_cols, bool fp4) {
    const auto r = r_in.value_or(torch::Tensor()), x0 = x0_in.value_or(torch::Tensor());
    const auto lr = lr_in.value_or(torch::Tensor()), l0 = l0_in.value_or(torch::Tensor());
    check_bf16(x, x);
    const int64_t cols = x.size(-1), rows = x.numel() / cols;
    TORCH_CHECK(cols % 8 == 0 && cols <= kernels::kResidualNormMaxCols, "last dim must be a multiple of 8, <= 2048");
    const bool add = r.defined(), blend = x0.defined();
    if (add)
      check_bf16(r, x);
    if (blend) {
      check_bf16(x0, x);
      TORCH_CHECK(lr.scalar_type() == torch::kFloat32 && l0.scalar_type() == torch::kFloat32);
      TORCH_CHECK(
            layer >= 0 && layer < lr.numel() && lr.numel() == l0.numel() && lr.is_contiguous() && l0.is_contiguous());
    }
    auto res = torch::empty_like(x), n = torch::empty_like(x); // mx: n's columns [0, gate_cols) only
    auto rstd = torch::empty({rows}, x.options().dtype(torch::kFloat32));
    const auto s = add && blend ? torch::empty_like(x) : torch::Tensor();
    const kernels::ResidualNormFwd args{
          .x = x.data_ptr(),
          .r = ptr(r),
          .x0 = ptr(x0),
          .lr = blend ? lr.data_ptr<float>() + layer : nullptr,
          .l0 = blend ? l0.data_ptr<float>() + layer : nullptr,
          .s = s.defined() ? s.data_ptr() : nullptr,
          .res = res.data_ptr(),
          .n = n.data_ptr(),
          .rstd = rstd.data_ptr<float>(),
          .rows = rows,
          .cols = static_cast<int>(cols),
          .eps = std::numeric_limits<float>::epsilon()}; // F.rms_norm's default eps (opmath float epsilon)
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    Fp8Tensor n_mx;
    if (mx) {
      TORCH_CHECK(residual_norm_mx_fits(rows, cols) && gate_cols >= 0 && gate_cols <= cols, "residual_norm_mx: shape");
      n_mx = empty_mx(
            rows, cols, x.options(), true, true, fp4 ? Nvfp4Role::FwdInput : Nvfp4Role::None, Nvfp4Role::WgradInput);
      const auto [out, out_t] = mx_outs(n_mx);
      kernels::residual_norm_mx_fwd(
            {.base = args, .out = out, .out_t = out_t, .n_cols = static_cast<int>(gate_cols)}, stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      finish_fp4(n_mx);
    }
    else
      kernels::residual_norm_fwd(args, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ctx->set_materialize_grads(false);
    ctx->save_for_backward({blend ? (add ? s : x) : torch::Tensor(), x0, lr, l0, res, rstd});
    ctx->saved_data["layer"] = layer;
    ctx->saved_data["add"] = add;
    if (x0_grad) {
      TORCH_CHECK(blend, "x0_grad needs x0");
      ctx->saved_data["x0_grad"] = c10::IValue::make_capsule(x0_grad);
      ctx->saved_data["x0_grad_owner"] = !x0_grad->claimed;
      ctx->saved_data["x_is_x0"] = x.is_same(x0);
      x0_grad->claimed = true;
    }
    if (!mx)
      return {res, n};
    // amax_t, amax: NVFP4 data_t's, data's (else placeholders)
    const auto amax_t = n_mx.fp4_t() ? n_mx.amax_t : torch::empty({0}, rstd.options());
    const auto amax = n_mx.fp4() ? n_mx.amax : torch::empty({0}, rstd.options());
    ctx->mark_non_differentiable({n_mx.data, n_mx.inv_scale, n_mx.data_t, n_mx.inv_scale_t, amax_t, amax});
    return {res, n, n_mx.data, n_mx.inv_scale, n_mx.data_t, n_mx.inv_scale_t, amax_t, amax};
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto saved = ctx->get_saved_variables();
    const auto &s = saved[0], &x0 = saved[1], &lr = saved[2], &l0 = saved[3], &res = saved[4], &rstd = saved[5];
    const auto layer = ctx->saved_data["layer"].toInt();
    const bool add = ctx->saved_data["add"].toBool(), blend = x0.defined();
    const auto g_res = grads[0].defined() ? grads[0].contiguous() : torch::Tensor();
    const auto g_n = grads[1].defined() ? grads[1].contiguous() : torch::Tensor();
    for (const auto& g : {g_res, g_n})
      if (g.defined())
        check_bf16(g, res);
    c10::intrusive_ptr<X0Grad> x0_grad;
    bool owner = false, fold_x = false, sum = false;
    if (ctx->saved_data.count("x0_grad") != 0) {
      x0_grad = c10::static_intrusive_pointer_cast<X0Grad>(ctx->saved_data["x0_grad"].toCapsule());
      owner = ctx->saved_data["x0_grad_owner"].toBool();
      fold_x = owner && ctx->saved_data["x_is_x0"].toBool();
    }
    const auto ds = fold_x && !add ? torch::Tensor() : torch::empty_like(res);
    torch::Tensor dx0, dlr, dl0, partials;
    if (blend) {
      if (x0_grad) {
        sum = x0_grad->sum.defined();
        if (!sum)
          x0_grad->sum = torch::empty_like(x0);
        dx0 = x0_grad->sum;
      }
      else
        dx0 = torch::empty_like(x0);
      dlr = torch::empty_like(lr);
      dl0 = torch::empty_like(l0);
      partials = torch::empty({2 * kernels::residual_norm_bwd_blocks(rstd.numel())}, lr.options());
    }
    kernels::residual_norm_bwd(
          {.g_res = ptr(g_res),
           .g_n = ptr(g_n),
           .res = res.data_ptr(),
           .rstd = rstd.data_ptr<float>(),
           .s = ptr(s),
           .x0 = ptr(x0),
           .lr = blend ? lr.data_ptr<float>() + layer : nullptr,
           .l0 = blend ? l0.data_ptr<float>() + layer : nullptr,
           .ds = ds.defined() ? ds.data_ptr() : nullptr,
           .dx0 = blend ? dx0.data_ptr() : nullptr,
           .dx0_sum = sum,
           .dx0_add_ds = fold_x,
           .partials = blend ? partials.data_ptr<float>() : nullptr,
           .dlr = blend ? dlr.data_ptr<float>() : nullptr,
           .dl0 = blend ? dl0.data_ptr<float>() : nullptr,
           .layer = static_cast<int>(layer),
           .n_layer = blend ? static_cast<int>(lr.numel()) : 0,
           .rows = rstd.numel(),
           .cols = static_cast<int>(res.size(-1))},
          at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (x0_grad) {
      if (!owner)
        dx0 = torch::Tensor();
      else
        x0_grad->sum = torch::Tensor(); // a retained graph's next backward starts over
      if (fold_x)
        return {dx0, add ? ds : torch::Tensor(), {}, dlr, dl0, {}, {}, {}, {}, {}};
    }
    // the add passes ds to both of its inputs
    return {ds, add ? ds : torch::Tensor(), dx0, dlr, dl0, {}, {}, {}, {}, {}};
  }
};

} // namespace

std::pair<torch::Tensor, torch::Tensor> residual_norm(
      const torch::Tensor& x, const torch::Tensor& r, const torch::Tensor& x0, const torch::Tensor& resid_lambdas,
      const torch::Tensor& x0_lambdas, int64_t layer, const c10::intrusive_ptr<X0Grad>& x0_grad) {
  const auto out = ResidualNorm::apply(
        x, optional(r), optional(x0), optional(resid_lambdas), optional(x0_lambdas), layer, x0_grad, false, 0, false);
  return {out[0], out[1]};
}

ResidualNormMx residual_norm_mx(
      const torch::Tensor& x, const torch::Tensor& r, const torch::Tensor& x0, const torch::Tensor& resid_lambdas,
      const torch::Tensor& x0_lambdas, int64_t layer, const c10::intrusive_ptr<X0Grad>& x0_grad, int64_t gate_cols,
      bool fp4) {
  const auto out = ResidualNorm::apply(
        x, optional(r), optional(x0), optional(resid_lambdas), optional(x0_lambdas), layer, x0_grad, true, gate_cols,
        fp4);
  Fp8Tensor n_mx{out[2], out[4], out[3], out[5]};
  if (n_mx.fp4_t())
    n_mx.amax_t = out[6];
  if (n_mx.fp4())
    n_mx.amax = out[7];
  return {out[0], out[1], n_mx};
}

bool residual_norm_mx_fits(int64_t rows, int64_t cols) {
  static const int max_smem = [] {
    int device = 0, n = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&n, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    return n;
  }();
  return mx_fits(rows, cols) && cols <= kernels::kResidualNormMxMaxCols &&
         kernels::residual_norm_mx_smem(static_cast<int>(cols)) <= max_smem;
}

} // namespace nanochat
