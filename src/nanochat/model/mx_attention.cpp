#include "nanochat/model/mx_attention.h"

#include <limits>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/mx_attention_kernel.h"
#include "nanochat/model/rotary_norm_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

// autograd::Function inputs can't be undefined tensors, but can be empty optionals
using Optional = std::optional<torch::Tensor>;

void* offset(const torch::Tensor& t, int64_t elements) {
  return static_cast<char*>(t.data_ptr()) + elements * t.element_size();
}

class MxAttentionInputs : public torch::autograd::Function<MxAttentionInputs> {
public:
  static variable_list forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& wq, const torch::Tensor& wk,
        const torch::Tensor& wv, const torch::Tensor& cos, const torch::Tensor& sin, double scale,
        const Optional& ve_in, const Optional& w_gate_in, int64_t head_dim, Fp8WeightCache* cache,
        const Fp8Tensor* x_mx) {
    const auto ve = ve_in.value_or(torch::Tensor()), w_gate = w_gate_in.value_or(torch::Tensor());
    const int64_t N = x.size(0), T = cos.size(1), nq = wq.size(0), nkv = wk.size(0), n = nq + 2 * nkv;
    TORCH_CHECK(mx_attention_fits(N, x.size(1), nq, nkv, head_dim) && wv.size(0) == nkv && N % T == 0);
    for (const auto& t : {cos, sin})
      TORCH_CHECK(
            t.scalar_type() == torch::kBFloat16 && t.dim() == 4 && t.size(3) == head_dim / 2 && t.stride(3) == 1 &&
                  t.stride(1) == head_dim / 2,
            "cos, sin: (1, T, 1, head_dim / 2) bf16 with contiguous rows");
    const int heads = static_cast<int>(nq / head_dim), kv_heads = static_cast<int>(nkv / head_dim);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(x_mx == nullptr || (x_mx->data.defined() && x_mx->data_t.defined()), "x_mx: both layouts");
    const auto xq = x_mx != nullptr ? *x_mx : quantize_mx(x);
    const auto wf = mx_qkv_weights(wq, wk, wv, cache);
    const auto qkv = at::_scaled_mm(xq.data, wf[0].t(), xq.inv_scale, wf[2], {}, {}, x.scalar_type(), false);

    // rotary + QK norm of q, k (read from qkv), as rotary_rms_norm
    const auto eps = std::numeric_limits<float>::epsilon();
    auto q = torch::empty({N, nq}, x.options()), k = torch::empty({N, nkv}, x.options());
    auto rstd_q = torch::empty({N * heads}, x.options().dtype(torch::kFloat32));
    auto rstd_k = torch::empty({N * kv_heads}, rstd_q.options());
    kernels::rotary_norm_fwd(
          qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), q.data_ptr(), rstd_q.data_ptr<float>(), rstd_q.numel(), heads,
          T, static_cast<int>(head_dim), n, static_cast<float>(scale), eps, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    kernels::rotary_norm_fwd(
          offset(qkv, nq), cos.data_ptr(), sin.data_ptr(), k.data_ptr(), rstd_k.data_ptr<float>(), rstd_k.numel(),
          kv_heads, T, static_cast<int>(head_dim), n, static_cast<float>(scale), eps, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // value embedding mix, the gate as gpt.py's bf16 Linear
    torch::Tensor v, gate_in, w_gate_bf16, z;
    if (ve.defined()) {
      TORCH_CHECK(
            ve.scalar_type() == torch::kBFloat16 && ve.is_contiguous() && ve.size(0) == N && ve.size(1) == nkv &&
            w_gate.size(0) == kv_heads);
      gate_in = x.narrow(1, 0, w_gate.size(1)).contiguous();
      w_gate_bf16 = w_gate.to(x.scalar_type());
      z = at::mm(gate_in, w_gate_bf16.t());
      v = torch::empty({N, nkv}, x.options());
      kernels::value_mix_fwd(offset(qkv, nq + nkv), n, z.data_ptr(), ve.data_ptr(), v.data_ptr(), N, kv_heads, stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else
      v = qkv.narrow(1, nq + nkv, nkv);

    ctx->save_for_backward(
          {xq.data_t, xq.inv_scale_t, wf[1], wf[3], qkv, rstd_q, rstd_k, cos, sin, gate_in, w_gate_bf16, z, ve});
    if (mx_grad_direct(wq) && mx_grad_direct(wk) && mx_grad_direct(wv))
      ctx->saved_data["weights"] = std::vector{wq, wk, wv}; // backward writes their .grad itself
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["head_dim"] = head_dim;
    ctx->saved_data["w_gate_dtype"] = static_cast<int64_t>(w_gate.defined() ? w_gate.scalar_type() : x.scalar_type());
    return {q, k, v};
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto s = ctx->get_saved_variables();
    const auto &x_t = s[0], &x_scale_t = s[1], &w_t = s[2], &w_scale_t = s[3], &qkv = s[4];
    const auto &rstd_q = s[5], &rstd_k = s[6], &cos = s[7], &sin = s[8];
    const auto &gate_in = s[9], &w_gate_bf16 = s[10], &z = s[11], &ve = s[12];
    const auto scale = static_cast<float>(ctx->saved_data["scale"].toDouble());
    const auto head_dim = ctx->saved_data["head_dim"].toInt();
    const int64_t N = x_t.size(1), T = cos.size(1), n = qkv.size(1), nq = grads[0].size(1), nkv = grads[1].size(1);
    const int heads = static_cast<int>(nq / head_dim), kv_heads = static_cast<int>(nkv / head_dim);
    const auto dtype = grads[0].scalar_type();
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    // dq, dk, dv, MX-quantized into the merged gradient (N, n) and its transpose
    const auto e4m3 = qkv.options().dtype(torch::kFloat8_e4m3fn);
    auto g = torch::empty({N, n}, e4m3), g_t = torch::empty({n, N}, e4m3);
    auto g_scale = empty_mx_scale(N, n, qkv.options()), g_scale_t = empty_mx_scale(n, N, qkv.options());
    const auto part = [&](int64_t col) {
      return std::pair{mx_out(g, g_scale, 0, col), mx_out(g_t, g_scale_t, col, 0)};
    };
    const auto dq = grads[0].contiguous(), dk = grads[1].contiguous(), dv = grads[2].contiguous();
    for (const auto& [d, col, rstd, h] :
         {std::tuple{dq, int64_t{0}, rstd_q, heads}, std::tuple{dk, nq, rstd_k, kv_heads}}) {
      const auto [out, out_t] = part(col);
      kernels::rotary_norm_bwd_mx(
            d.data_ptr(), offset(qkv, col), cos.data_ptr(), sin.data_ptr(), rstd.data_ptr<float>(), N, h, T, n, scale,
            out, out_t, stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    torch::Tensor dve, dz;
    const auto [v_out, v_out_t] = part(nq + nkv);
    if (ve.defined()) {
      dve = torch::empty_like(ve);
      dz = torch::empty_like(z);
      kernels::value_mix_bwd_mx(
            dv.data_ptr(), z.data_ptr(), ve.data_ptr(), dve.data_ptr(), dz.data_ptr(), N, kv_heads, v_out, v_out_t,
            stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else
      quantize_mx_into(dv, false, v_out, v_out_t);

    auto dx = at::_scaled_mm(g, w_t.t(), g_scale, w_scale_t, {}, {}, dtype, false);
    torch::Tensor dw;
    if (ctx->saved_data.count("weights") != 0)
      mx_grad_weights(g_t, g_scale_t, x_t, x_scale_t, ctx->saved_data["weights"].toTensorVector());
    else
      dw = at::_scaled_mm(g_t, x_t.t(), g_scale_t, x_scale_t, {}, {}, dtype, false);
    const auto rows = [&](int64_t row, int64_t size) {
      return dw.defined() ? dw.narrow(0, row, size) : torch::Tensor();
    };
    torch::Tensor dw_gate;
    if (ve.defined()) { // the gate Linear's backward, as autograd's mm
      dx.narrow(1, 0, gate_in.size(1)).add_(at::mm(dz, w_gate_bf16));
      dw_gate = at::mm(dz.t(), gate_in).to(static_cast<torch::ScalarType>(ctx->saved_data["w_gate_dtype"].toInt()));
    }
    return {dx, rows(0, nq), rows(nq, nkv), rows(nq + nkv, nkv), {}, {}, {}, dve, dw_gate, {}, {}, {}};
  }
};

} // namespace

bool mx_attention_fits(int64_t N, int64_t C, int64_t n_q, int64_t n_kv, int64_t head_dim) {
  return head_dim == kernels::kMxHeadDim && mx_fits(N, C) && mx_fits(n_q, n_kv);
}

variable_list mx_attention_inputs(
      const torch::Tensor& x_2d, const torch::Tensor& wq, const torch::Tensor& wk, const torch::Tensor& wv,
      const torch::Tensor& cos, const torch::Tensor& sin, double scale, const torch::Tensor& ve,
      const torch::Tensor& w_gate, int64_t head_dim, Fp8WeightCache* cache, const Fp8Tensor* x_mx) {
  const auto opt = [](const torch::Tensor& t) {
    return t.defined() ? Optional(t) : std::nullopt;
  };
  return MxAttentionInputs::apply(
        x_2d.contiguous(), wq, wk, wv, cos, sin, scale, opt(ve), opt(w_gate), head_dim, cache, x_mx);
}

} // namespace nanochat
