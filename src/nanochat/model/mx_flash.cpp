#include "nanochat/model/mx_flash.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/mx_flash_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

constexpr int64_t kD = kernels::kMxFlashHeadDim;

// (B, T, heads, 128) bf16 with contiguous heads; returns the token stride in elements
int64_t check_input(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 4, name, " must be (B, T, H, D) bf16");
  TORCH_CHECK(x.size(3) == kD, name, ": head_dim must be ", kD);
  TORCH_CHECK(x.size(1) % kernels::kMxFlashBlockM == 0, name, ": T must divide by ", kernels::kMxFlashBlockM);
  TORCH_CHECK(
        x.stride(3) == 1 && x.stride(2) == kD && x.stride(0) == x.size(1) * x.stride(1), name,
        ": heads must be contiguous within a token, tokens evenly strided");
  return x.stride(1);
}

std::optional<int64_t> left_window(int64_t window, int64_t T) {
  return window >= 0 && window < T ? std::optional(window) : std::nullopt;
}

class MxFlashAttention : public torch::autograd::Function<MxFlashAttention> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
        const MxFlashInputs* pre) {
    auto [out, lse] = mx_flash_forward(q, k, v, window, pre);
    ctx->save_for_backward({q, k, v, out, lse});
    ctx->saved_data["window"] = window;
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto saved = ctx->get_saved_variables();
    const auto &q = saved[0], &k = saved[1], &v = saved[2], &out = saved[3], &lse = saved[4];
    const int64_t T = q.size(1);
    const auto window = left_window(ctx->saved_data["window"].toInt(), T);
    const auto empty = torch::empty({0}, q.options().dtype(torch::kUInt64));
    auto [dq, dk, dv] = at::_flash_attention_backward(
          grads[0].contiguous(), q, k, v, out, lse, torch::Tensor(), torch::Tensor(), T, T, 0.0, /*is_causal=*/true,
          empty, empty, std::nullopt, window, window ? std::optional<int64_t>(0) : std::nullopt);
    return {dq, dk, dv, torch::Tensor(), torch::Tensor()};
  }
};

} // namespace

std::pair<torch::Tensor, torch::Tensor> mx_flash_quantize_rows(const torch::Tensor& x) {
  const int64_t stride = check_input(x, "x");
  const int64_t B = x.size(0), T = x.size(1), heads = x.size(2);
  auto data = torch::empty({B, T, heads, kD}, x.options().dtype(torch::kUInt8));
  auto scale = torch::empty({B, heads, T}, x.options().dtype(torch::kInt32));
  kernels::mx_flash_quantize_rows(
        x.data_ptr(), data.data_ptr(), reinterpret_cast<uint32_t*>(scale.data_ptr<int32_t>()), static_cast<int>(B), T,
        static_cast<int>(heads), stride, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {data, scale};
}

std::pair<torch::Tensor, torch::Tensor> mx_flash_quantize_vt(const torch::Tensor& v) {
  const int64_t stride = check_input(v, "v");
  const int64_t B = v.size(0), T = v.size(1), heads = v.size(2);
  auto data = torch::empty({B, heads, kD, T}, v.options().dtype(torch::kUInt8));
  auto scale = torch::empty({B, heads, T / 64, kD, 2}, v.options().dtype(torch::kUInt8));
  kernels::mx_flash_quantize_vt(
        v.data_ptr(), data.data_ptr(), scale.data_ptr<uint8_t>(), static_cast<int>(B), T, static_cast<int>(heads),
        stride, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {data, scale};
}

std::pair<torch::Tensor, torch::Tensor> mx_flash_forward(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
      const MxFlashInputs* pre) {
  check_input(q, "q");
  check_input(k, "k");
  check_input(v, "v");
  const int64_t B = q.size(0), T = q.size(1), H = q.size(2), Hkv = k.size(2);
  TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) == B && k.size(1) == T, "k, v must be (B, T, Hkv, D)");
  TORCH_CHECK(H % Hkv == 0, "H must divide by Hkv");
  MxFlashInputs in;
  if (pre != nullptr) {
    in = *pre;
    const auto u8 = torch::kUInt8;
    const bool ok = in.q.sizes() == q.sizes() && in.k.sizes() == k.sizes() && in.q.scalar_type() == u8 &&
                    in.k.scalar_type() == u8 && in.q_scale.sizes() == torch::IntArrayRef{B, H, T} &&
                    in.k_scale.sizes() == torch::IntArrayRef{B, Hkv, T} &&
                    in.vt.sizes() == torch::IntArrayRef{B, Hkv, kD, T} && in.vt.scalar_type() == u8 &&
                    in.v_scale.numel() == B * Hkv * T * 4;
    TORCH_CHECK(ok, "pre: mx_flash_quantize_rows / mx_flash_quantize_vt layouts");
    for (const auto* t : {&in.q, &in.q_scale, &in.k, &in.k_scale, &in.vt, &in.v_scale})
      TORCH_CHECK(t->is_contiguous(), "pre: contiguous");
  }
  else {
    std::tie(in.q, in.q_scale) = mx_flash_quantize_rows(q);
    std::tie(in.k, in.k_scale) = mx_flash_quantize_rows(k);
    std::tie(in.vt, in.v_scale) = mx_flash_quantize_vt(v);
  }
  const auto &qd = in.q, &qs = in.q_scale, &kd = in.k, &ks = in.k_scale, &vd = in.vt, &vs = in.v_scale;
  auto out = torch::empty({B, T, H, kD}, q.options());
  auto lse = torch::empty({B, H, T}, q.options().dtype(torch::kFloat32));
  kernels::mx_flash_fwd(
        qd.data_ptr(), reinterpret_cast<const uint32_t*>(qs.data_ptr<int32_t>()), kd.data_ptr(),
        reinterpret_cast<const uint32_t*>(ks.data_ptr<int32_t>()), vd.data_ptr(), vs.data_ptr<uint8_t>(),
        out.data_ptr(), lse.data_ptr<float>(), static_cast<int>(B), T, static_cast<int>(H), static_cast<int>(Hkv),
        window, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

torch::Tensor mx_flash_attention(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
      const MxFlashInputs* pre) {
  return MxFlashAttention::apply(q, k, v, window, pre);
}

} // namespace nanochat
