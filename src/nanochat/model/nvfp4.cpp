#include "nanochat/model/nvfp4.h"

#include <atomic>
#include <map>
#include <mutex>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/nvfp4_gemm_kernel.h"
#include "nanochat/model/nvfp4_kernel.h"

namespace nanochat {

namespace {

std::atomic<const Nvfp4Backward*> backward_options{nullptr};
std::atomic<uint64_t> seed_counter{1};

cudaStream_t stream() {
  return at::cuda::getCurrentCUDAStream().stream();
}

void check_mx(const torch::Tensor& data, const torch::Tensor& scale) {
  TORCH_CHECK(
        data.is_cuda() && data.dim() == 2 && data.is_contiguous() && data.scalar_type() == torch::kFloat8_e4m3fn &&
              data.size(0) % 128 == 0 && data.size(1) % 128 == 0 && scale.numel() * 32 == data.numel(),
        "NVFP4: expected MX data (R, C) e4m3, dims % 128, with its scales");
}

// stochastic rounding's: a fresh one per tensor
uint64_t next_seed(const Nvfp4Backward& o) {
  return seed_counter++ + (o.seed << 40);
}

} // namespace

const torch::Tensor& nvfp4_hadamard(const torch::Device& device) {
  static std::mutex mutex;
  static std::map<int, torch::Tensor> cache;
  const std::lock_guard lock(mutex);
  auto& h = cache[device.index()];
  if (!h.defined()) {
    auto m = torch::ones({1, 1});
    for (int n = 1; n < 16; n *= 2)
      m = torch::cat({torch::cat({m, m}, 1), torch::cat({m, -m}, 1)}, 0);
    auto gen = at::make_generator<at::CPUGeneratorImpl>(0);
    const auto signs = torch::randint(0, 2, {16, 1}, gen, torch::kFloat32) * 2 - 1;
    h = (signs * m / 4).contiguous().to(device);
  }
  return h;
}

uint32_t nvfp4_hadamard_signs() {
  static const uint32_t signs = [] {
    const auto h = nvfp4_hadamard(torch::kCPU);
    uint32_t bits = 0;
    for (int i = 0; i < 16; ++i)
      bits |= h[i][0].item<float>() < 0.f ? 1u << i : 0u;
    return bits;
  }();
  return signs;
}

Nvfp4Target empty_nvfp4(
      int64_t R, int64_t C, const torch::TensorOptions& options, bool rht, bool stochastic, uint64_t seed) {
  TORCH_CHECK(R % 128 == 0 && C % 64 == 0, "empty_nvfp4: R % 128, C % 64");
  auto smax = torch::empty({kernels::kNvfp4Slots}, options.dtype(torch::kInt32));
  C10_CUDA_CHECK(cudaMemsetAsync(smax.data_ptr(), 0, smax.nbytes(), stream()));
  return {
        torch::empty({R, C / 2}, options.dtype(torch::kUInt8)),
        torch::empty({R * C / 16}, options.dtype(torch::kInt16)),
        smax,
        rht,
        stochastic,
        seed};
}

kernels::Nvfp4Out nvfp4_out(const Nvfp4Target& t, int64_t row, int64_t col) {
  TORCH_CHECK(row % 128 == 0 && col % 64 == 0, "nvfp4_out: row % 128, col % 64");
  const int64_t ld = t.data.size(1) * 2, i = row * ld + col;
  return {
        static_cast<uint8_t*>(t.data.data_ptr()) + i / 2,
        ld,
        static_cast<int16_t*>(t.scale16.data_ptr()) + ((row / 128) * (ld / 64) + col / 64) * 512,
        reinterpret_cast<unsigned*>(t.smax.data_ptr<int32_t>()),
        i,
        t.seed,
        nvfp4_hadamard_signs(),
        t.rht,
        t.stochastic};
}

namespace {

// t's output tensors and the kernel's view of both
std::pair<Nvfp4Tensor, kernels::Nvfp4Finish> finish_args(const Nvfp4Target& t) {
  const int64_t R = t.data.size(0), C = t.data.size(1) * 2;
  Nvfp4Tensor out{
        t.data, torch::empty({R * C / 16}, t.data.options()),
        torch::empty({}, t.data.options().dtype(torch::kFloat32))};
  const kernels::Nvfp4Finish args{t.scale16.data_ptr(),
                                  reinterpret_cast<const unsigned*>(t.smax.data_ptr<int32_t>()),
                                  R,
                                  C,
                                  out.scale.data_ptr(),
                                  out.amax.data_ptr<float>()};
  return {out, args};
}

} // namespace

Nvfp4Tensor nvfp4_finish(const Nvfp4Target& t) {
  const auto [out, args] = finish_args(t);
  kernels::nvfp4_finish(args, nullptr, stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::pair<Nvfp4Tensor, Nvfp4Tensor> nvfp4_finish(const Nvfp4Target& a, const Nvfp4Target& b) {
  const auto [out_a, args_a] = finish_args(a);
  const auto [out_b, args_b] = finish_args(b);
  kernels::nvfp4_finish(args_a, &args_b, stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_a, out_b};
}

Nvfp4Tensor mx_to_nvfp4(
      const torch::Tensor& data, const torch::Tensor& scale, bool rht, bool stochastic, uint64_t seed) {
  check_mx(data, scale);
  const int64_t R = data.size(0), C = data.size(1);
  const auto u8 = data.options().dtype(torch::kUInt8);
  Nvfp4Tensor t{
        torch::empty({R, C / 2}, u8), torch::empty({R * C / 16}, u8),
        torch::zeros({}, data.options().dtype(torch::kFloat32))};
  kernels::mx_to_nvfp4(
        data.data_ptr(), scale.data_ptr(), R, C, rht ? nvfp4_hadamard(data.device()).data_ptr<float>() : nullptr,
        stochastic, seed, t.data.data_ptr(), t.scale.data_ptr(), t.amax.data_ptr<float>(), stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return t;
}

torch::Tensor nvfp4_to_bf16(const Nvfp4Tensor& t) {
  const int64_t R = t.data.size(0), C = t.data.size(1) * 2;
  auto out = torch::empty({R, C}, t.data.options().dtype(torch::kBFloat16));
  kernels::nvfp4_to_bf16(
        t.data.data_ptr(), t.scale.data_ptr(), t.amax.data_ptr<float>(), R, C, out.data_ptr(), stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor mx_to_bf16(const torch::Tensor& data, const torch::Tensor& scale) {
  check_mx(data, scale);
  auto out = torch::empty(data.sizes(), data.options().dtype(torch::kBFloat16));
  kernels::mx_to_bf16(data.data_ptr(), scale.data_ptr(), data.size(0), data.size(1), out.data_ptr(), stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void nvfp4_gemm_f32(
      const Nvfp4Tensor& a, const Nvfp4Tensor& b, const torch::Tensor& out, bool accumulate,
      const torch::Tensor& alpha) {
  const int64_t M = a.data.size(0), N = b.data.size(0), K = a.data.size(1) * 2;
  TORCH_CHECK(
        b.data.size(1) * 2 == K && out.is_contiguous() && out.scalar_type() == torch::kFloat32 && out.size(0) == M &&
              out.size(1) == N,
        "nvfp4_gemm_f32: shape mismatch");
  TORCH_CHECK(!alpha.defined() || (alpha.numel() == 1 && alpha.scalar_type() == torch::kFloat32));
  const char* error = kernels::cutlass_nvfp4_gemm_f32(
        a.data.data_ptr(), a.scale.data_ptr(), b.data.data_ptr(), b.scale.data_ptr(), out.data_ptr<float>(), M, N, K,
        a.amax.data_ptr<float>(), b.amax.data_ptr<float>(), alpha.defined() ? alpha.data_ptr<float>() : nullptr,
        accumulate, stream());
  TORCH_CHECK(error == nullptr, error);
}

torch::Tensor nvfp4_gemm(const Nvfp4Tensor& a, const Nvfp4Tensor& b) {
  const int64_t M = a.data.size(0), N = b.data.size(0), K = a.data.size(1) * 2;
  TORCH_CHECK(b.data.size(1) * 2 == K, "nvfp4_gemm: shape mismatch");
  auto out = torch::empty({M, N}, a.data.options().dtype(torch::kBFloat16));
  const char* error = kernels::cutlass_nvfp4_gemm_bf16(
        a.data.data_ptr(), a.scale.data_ptr(), b.data.data_ptr(), b.scale.data_ptr(), out.data_ptr(), M, N, K,
        a.amax.data_ptr<float>(), b.amax.data_ptr<float>(), stream());
  TORCH_CHECK(error == nullptr, error);
  return out;
}

void set_nvfp4_backward(const Nvfp4Backward* options) {
  backward_options = options;
}

const Nvfp4Backward* nvfp4_backward() {
  return backward_options;
}

Nvfp4Target nvfp4_target(int64_t R, int64_t C, const torch::TensorOptions& options, Nvfp4Role role) {
  const auto* o = nvfp4_backward();
  if (o == nullptr || role == Nvfp4Role::None)
    return {};
  const bool wgrad = role == Nvfp4Role::WgradInput || role == Nvfp4Role::WgradGrad;
  if (!(wgrad ? o->wgrad : o->dgrad))
    return {};
  const bool sr = role == Nvfp4Role::WgradGrad ? o->sr : role == Nvfp4Role::DgradGrad && o->sr_dgrad;
  return empty_nvfp4(R, C, options, wgrad && o->rht, sr, sr ? next_seed(*o) : 0);
}

void nvfp4_grad_weight(
      const torch::Tensor& go_t, const torch::Tensor& go_scale_t, const torch::Tensor& go_amax,
      const torch::Tensor& in_t, const torch::Tensor& in_scale_t, const torch::Tensor& in_amax,
      const torch::Tensor& out, bool accumulate, const torch::Tensor& alpha) {
  const auto operand = [](const torch::Tensor& data, const torch::Tensor& scale, const torch::Tensor& amax, bool grad) {
    if (amax.defined())
      return Nvfp4Tensor{data, scale, amax};
    const auto* o = nvfp4_backward();
    TORCH_CHECK(o != nullptr && o->wgrad, "nvfp4_grad_weight: MX operands need NVFP4 weight gradients on");
    return mx_to_nvfp4(data, scale, o->rht, grad && o->sr, grad && o->sr ? next_seed(*o) : 0);
  };
  nvfp4_gemm_f32(
        operand(go_t, go_scale_t, go_amax, true), operand(in_t, in_scale_t, in_amax, false), out, accumulate, alpha);
}

} // namespace nanochat
