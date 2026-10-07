#include "nanochat/model/attention/mx_flash.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/attention/flash_kernel.h"
#include "nanochat/model/attention/mx_flash_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

constexpr int64_t kD = kernels::kMxFlashHeadDim;

// (B, T, heads, 128) bf16 with contiguous heads, 16-byte aligned tokens; returns the token stride in elements
int64_t check_input(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 4, name, " must be (B, T, H, D) bf16");
  TORCH_CHECK(x.size(3) == kD, name, ": head_dim must be ", kD);
  TORCH_CHECK(x.size(1) % kernels::kMxFlashBlockM == 0, name, ": T must divide by ", kernels::kMxFlashBlockM);
  TORCH_CHECK(
        x.stride(3) == 1 && x.stride(2) == kD && x.stride(0) == x.size(1) * x.stride(1), name,
        ": heads must be contiguous within a token, tokens evenly strided");
  TORCH_CHECK(x.stride(1) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
  return x.stride(1);
}

// in's tensors for (B, T, H) queries and Hkv keys: the forward's (q, k, vt) and/or the backward's (all but vt)
void check_quantized(const MxFlashInputs& in, int64_t B, int64_t T, int64_t H, int64_t Hkv, bool fwd, bool bwd) {
  const auto check = [&](const torch::Tensor& t, torch::IntArrayRef sizes, torch::ScalarType dtype, const char* name) {
    TORCH_CHECK(
          t.defined() && t.is_cuda() && t.sizes() == sizes && t.scalar_type() == dtype && t.is_contiguous(),
          "MxFlashInputs.", name, ": expected ", sizes, " ", dtype, ", contiguous");
  };
  const auto rows = [&](const torch::Tensor& d, const torch::Tensor& s, int64_t heads, const char* name) {
    check(d, {B, T, heads, kD}, torch::kUInt8, name);
    check(s, {B, heads, T}, torch::kInt32, name);
  };
  const auto t = [&](const torch::Tensor& d, const torch::Tensor& s, int64_t heads, const char* name) {
    check(d, {B, heads, kD, T}, torch::kUInt8, name);
    check(s, {B, heads, T / 32, kD}, torch::kUInt8, name);
  };
  rows(in.q, in.q_scale, H, "q");
  rows(in.k, in.k_scale, Hkv, "k");
  if (fwd)
    t(in.vt, in.vt_scale, Hkv, "vt");
  if (bwd) {
    rows(in.v, in.v_scale, Hkv, "v");
    t(in.qt, in.qt_scale, H, "qt");
    t(in.kt, in.kt_scale, Hkv, "kt");
  }
}

const uint32_t* words(const torch::Tensor& t) {
  return reinterpret_cast<const uint32_t*>(t.data_ptr<int32_t>());
}

class MxFlashAttention : public torch::autograd::Function<MxFlashAttention> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
        const MxFlashInputs* pre) {
    const auto in = pre != nullptr ? *pre : mx_flash_quantize(q, k, v);
    check_quantized(in, q.size(0), q.size(1), q.size(2), k.size(2), true, true);
    auto [out, lse] = mx_flash_forward(q, k, v, window, &in);
    ctx->save_for_backward(
          {in.q, in.q_scale, in.k, in.k_scale, in.v, in.v_scale, in.qt, in.qt_scale, in.kt, in.kt_scale, out, lse});
    ctx->saved_data["window"] = window;
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto s = ctx->get_saved_variables();
    const MxFlashInputs in{s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], {}, {}};
    auto [dq, dk, dv] = mx_flash_backward(grads[0], in, s[10], s[11], ctx->saved_data["window"].toInt());
    return {dq, dk, dv, torch::Tensor(), torch::Tensor()};
  }
};

} // namespace

std::pair<torch::Tensor, torch::Tensor> mx_flash_quantize_rows(const torch::Tensor& x) {
  const int64_t ld = check_input(x, "x");
  const int64_t B = x.size(0), T = x.size(1), heads = x.size(2);
  auto data = torch::empty({B, T, heads, kD}, x.options().dtype(torch::kUInt8));
  auto scale = torch::empty({B, heads, T}, x.options().dtype(torch::kInt32));
  kernels::mx_flash_quantize_rows(
        x.data_ptr(), ld, data.data_ptr(), reinterpret_cast<uint32_t*>(scale.data_ptr<int32_t>()), nullptr, nullptr,
        static_cast<int>(B), T, static_cast<int>(heads), at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {data, scale};
}

std::pair<torch::Tensor, torch::Tensor> mx_flash_quantize_t(const torch::Tensor& x) {
  const int64_t ld = check_input(x, "x");
  const int64_t B = x.size(0), T = x.size(1), heads = x.size(2);
  auto data = torch::empty({B, heads, kD, T}, x.options().dtype(torch::kUInt8));
  auto scale = torch::empty({B, heads, T / 32, kD}, x.options().dtype(torch::kUInt8));
  kernels::mx_flash_quantize_t(
        x.data_ptr(), ld, data.data_ptr(), scale.data_ptr<uint8_t>(), static_cast<int>(B), T, static_cast<int>(heads),
        at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {data, scale};
}

MxFlashInputs mx_flash_quantize(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  MxFlashInputs in;
  std::tie(in.q, in.q_scale) = mx_flash_quantize_rows(q);
  std::tie(in.k, in.k_scale) = mx_flash_quantize_rows(k);
  std::tie(in.v, in.v_scale) = mx_flash_quantize_rows(v);
  std::tie(in.qt, in.qt_scale) = mx_flash_quantize_t(q);
  std::tie(in.kt, in.kt_scale) = mx_flash_quantize_t(k);
  std::tie(in.vt, in.vt_scale) = mx_flash_quantize_t(v);
  return in;
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
  if (pre != nullptr)
    in = *pre;
  else {
    std::tie(in.q, in.q_scale) = mx_flash_quantize_rows(q);
    std::tie(in.k, in.k_scale) = mx_flash_quantize_rows(k);
    std::tie(in.vt, in.vt_scale) = mx_flash_quantize_t(v);
  }
  check_quantized(in, B, T, H, Hkv, true, false);
  auto out = torch::empty({B, T, H, kD}, q.options());
  auto lse = torch::empty({B, H, T}, q.options().dtype(torch::kFloat32));
  kernels::mx_flash_fwd(
        in.q.data_ptr(), words(in.q_scale), in.k.data_ptr(), words(in.k_scale), in.vt.data_ptr(),
        in.vt_scale.data_ptr<uint8_t>(), out.data_ptr(), lse.data_ptr<float>(), static_cast<int>(B), T,
        static_cast<int>(H), static_cast<int>(Hkv), window, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mx_flash_backward(
      const torch::Tensor& dout_in, const MxFlashInputs& in, const torch::Tensor& out, const torch::Tensor& lse,
      int64_t window, int dq_variant, int dkv_variant) {
  const auto dout = dout_in.contiguous();
  TORCH_CHECK(
        dout.dim() == 4 && dout.size(3) == kD && dout.scalar_type() == torch::kBFloat16, "dout: (B, T, H, D) bf16");
  const int64_t B = dout.size(0), T = dout.size(1), H = dout.size(2), Hkv = in.k.size(2);
  TORCH_CHECK(T % kernels::kMxFlashBlockM == 0, "T must divide by ", kernels::kMxFlashBlockM);
  check_quantized(in, B, T, H, Hkv, false, true);
  TORCH_CHECK(
        out.sizes() == dout.sizes() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16, "out: as dout");
  TORCH_CHECK(lse.sizes() == torch::IntArrayRef({B, H, T}) && lse.is_contiguous(), "lse: (B, H, T) fp32");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto u8 = dout.options().dtype(torch::kUInt8);
  auto do8 = torch::empty({B, T, H, kD}, u8), dos = torch::empty({B, H, T}, u8.dtype(torch::kInt32));
  auto dot = torch::empty({B, H, kD, T}, u8), dots = torch::empty({B, H, T / 32, kD}, u8);
  auto delta = torch::empty({B, H, T}, lse.options());
  kernels::mx_flash_quantize_dout(
        dout.data_ptr(), out.data_ptr(), do8.data_ptr(), reinterpret_cast<uint32_t*>(dos.data_ptr<int32_t>()),
        dot.data_ptr(), dots.data_ptr<uint8_t>(), delta.data_ptr<float>(), static_cast<int>(B), T, static_cast<int>(H),
        stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const kernels::MxFlashBwdInputs k_in{
        in.q.data_ptr(),
        in.k.data_ptr(),
        in.v.data_ptr(),
        do8.data_ptr(),
        words(in.q_scale),
        words(in.k_scale),
        words(in.v_scale),
        words(dos),
        in.qt.data_ptr(),
        in.kt.data_ptr(),
        dot.data_ptr(),
        in.qt_scale.data_ptr<uint8_t>(),
        in.kt_scale.data_ptr<uint8_t>(),
        dots.data_ptr<uint8_t>(),
        lse.data_ptr<float>(),
        delta.data_ptr<float>()};
  auto dq = torch::empty({B, T, H, kD}, dout.options());
  auto dk = torch::empty({B, T, Hkv, kD}, dout.options()), dv = torch::empty({B, T, Hkv, kD}, dout.options());
  kernels::mx_flash_bwd(
        k_in, dq.data_ptr(), dk.data_ptr(), dv.data_ptr(), static_cast<int>(B), T, static_cast<int>(H),
        static_cast<int>(Hkv), window, stream, dq_variant, dkv_variant);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}

torch::Tensor mx_flash_attention(
      const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
      const MxFlashInputs* pre) {
  return MxFlashAttention::apply(q, k, v, window, pre);
}

} // namespace nanochat
