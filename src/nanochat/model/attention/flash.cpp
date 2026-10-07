#include "nanochat/model/attention/flash.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/attention/flash_kernel.h"
#include "nanochat/model/attention/mx_flash.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

constexpr int64_t kD = kernels::kFlashHeadDim;

// (B, T, heads, 128) bf16 with contiguous heads; returns the token stride in elements
int64_t check_input(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 4, name, " must be (B, T, H, D) bf16");
  TORCH_CHECK(x.size(3) == kD, name, ": head_dim must be ", kD);
  TORCH_CHECK(x.size(1) % kernels::kFlashSeqMultiple == 0, name, ": T must divide by ", kernels::kFlashSeqMultiple);
  TORCH_CHECK(
        x.stride(3) == 1 && x.stride(2) == kD && x.stride(0) == x.size(1) * x.stride(1), name,
        ": heads must be contiguous within a token, tokens evenly strided");
  TORCH_CHECK(x.stride(1) % 8 == 0 && x.stride(0) % 8 == 0, name, ": tokens must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
  return x.stride(1);
}

class FlashAttention : public torch::autograd::Function<FlashAttention> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
        bool mx_backward) {
    auto [out, lse] = flash_forward(q, k, v, window);
    ctx->save_for_backward({q, k, v, out, lse});
    ctx->saved_data["window"] = window;
    ctx->saved_data["mx"] = mx_backward;
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto saved = ctx->get_saved_variables();
    const auto &q = saved[0], &k = saved[1], &v = saved[2], &out = saved[3], &lse = saved[4];
    const int64_t T = q.size(1), window = ctx->saved_data["window"].toInt();
    if (ctx->saved_data["mx"].toBool()) {
      auto [dq, dk, dv] = flash_backward_mx(grads[0], q, k, v, out, lse, window);
      return {dq, dk, dv, torch::Tensor(), torch::Tensor()};
    }
    // FA2's backward: ours (flash_backward) is not faster
    const auto left = window >= 0 && window < T ? std::optional(window) : std::nullopt;
    const auto empty = torch::empty({0}, q.options().dtype(torch::kUInt64)); // rng state, unused without dropout
    auto [dq, dk, dv] = at::_flash_attention_backward(
          grads[0].contiguous(), q, k, v, out, lse, torch::Tensor(), torch::Tensor(), T, T, 0.0, /*is_causal=*/true,
          empty, empty, std::nullopt, left, left ? std::optional<int64_t>(0) : std::nullopt);
    return {dq, dk, dv, torch::Tensor(), torch::Tensor()};
  }
};

} // namespace

bool flash_supported(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  const auto ok = [](const torch::Tensor& x) {
    return x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 4 && x.size(3) == kD &&
           x.size(1) % kernels::kFlashSeqMultiple == 0 && x.stride(3) == 1 && x.stride(2) == kD &&
           x.stride(0) == x.size(1) * x.stride(1) && x.stride(1) % 8 == 0 && x.stride(0) % 8 == 0 &&
           reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
  };
  return ok(q) && ok(k) && ok(v) && k.sizes() == v.sizes() && k.size(0) == q.size(0) && k.size(1) == q.size(1) &&
         q.size(2) % k.size(2) == 0;
}

std::pair<torch::Tensor, torch::Tensor> flash_forward(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window, int variant) {
  const int64_t q_ld = check_input(q, "q"), k_ld = check_input(k, "k"), v_ld = check_input(v, "v");
  const int64_t B = q.size(0), T = q.size(1), H = q.size(2), Hkv = k.size(2);
  TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) == B && k.size(1) == T, "k, v must be (B, T, Hkv, D)");
  TORCH_CHECK(H % Hkv == 0, "H must divide by Hkv");
  auto out = torch::empty({B, T, H, kD}, q.options());
  auto lse = torch::empty({B, H, T}, q.options().dtype(torch::kFloat32));
  kernels::flash_fwd(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), lse.data_ptr<float>(), static_cast<int>(B), T,
        static_cast<int>(H), static_cast<int>(Hkv), q_ld, k_ld, v_ld, window, at::cuda::getCurrentCUDAStream().stream(),
        variant);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_backward(
      const torch::Tensor& dout_in, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
      const torch::Tensor& out, const torch::Tensor& lse, int64_t window, int dq_variant, int dkv_variant) {
  const int64_t q_ld = check_input(q, "q"), k_ld = check_input(k, "k"), v_ld = check_input(v, "v");
  const int64_t B = q.size(0), T = q.size(1), H = q.size(2), Hkv = k.size(2);
  const auto dout = dout_in.contiguous();
  TORCH_CHECK(dout.sizes() == q.sizes() && dout.scalar_type() == torch::kBFloat16, "dout: as q");
  TORCH_CHECK(out.sizes() == q.sizes() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16, "out: as q");
  TORCH_CHECK(lse.sizes() == torch::IntArrayRef({B, H, T}) && lse.is_contiguous(), "lse: (B, H, T) fp32");
  auto dq = torch::empty({B, T, H, kD}, q.options());
  auto dk = torch::empty({B, T, Hkv, kD}, q.options()), dv = torch::empty({B, T, Hkv, kD}, q.options());
  auto delta = torch::empty({B, H, T}, lse.options());
  kernels::flash_bwd(
        dout.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), lse.data_ptr<float>(),
        delta.data_ptr<float>(), dq.data_ptr(), dk.data_ptr(), dv.data_ptr(), static_cast<int>(B), T,
        static_cast<int>(H), static_cast<int>(Hkv), q_ld, k_ld, v_ld, window, at::cuda::getCurrentCUDAStream().stream(),
        dq_variant, dkv_variant);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_backward_mx(
      const torch::Tensor& dout, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
      const torch::Tensor& out, const torch::Tensor& lse, int64_t window, int dq_variant, int dkv_variant) {
  MxFlashInputs in;
  std::tie(in.q, in.q_scale) = mx_flash_quantize_rows(q);
  std::tie(in.k, in.k_scale) = mx_flash_quantize_rows(k);
  std::tie(in.v, in.v_scale) = mx_flash_quantize_rows(v);
  std::tie(in.qt, in.qt_scale) = mx_flash_quantize_t(q);
  std::tie(in.kt, in.kt_scale) = mx_flash_quantize_t(k);
  return mx_flash_backward(dout, in, out, lse, window, dq_variant, dkv_variant);
}

torch::Tensor flash_attention(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window, bool mx_backward) {
  return FlashAttention::apply(q, k, v, window, mx_backward);
}

} // namespace nanochat
