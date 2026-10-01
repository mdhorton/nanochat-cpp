#include "nanochat/model/nvfp4.h"

#include <atomic>
#include <map>
#include <mutex>
#include <vector>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/mx_gemm.h"
#include "nanochat/model/nvfp4_gemm_kernel.h"
#include "nanochat/model/nvfp4_kernel.h"

namespace nanochat {

namespace {

std::atomic<const Nvfp4Backward*> backward_options{nullptr};
std::atomic<bool> four_six_fwd{false};
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

namespace {

// smax slot sets per device, zeroed once, each leased to one target at a time under a fresh epoch
constexpr int64_t kSmaxSets = 4096;

struct SmaxPool {
  torch::Tensor slots; // (kSmaxSets, kNvfp4Slots)
  std::vector<std::weak_ptr<void>> leases;
  std::vector<uint32_t> epochs;
  int64_t next = 0;
};

void lease_smax(Nvfp4Target& t, const torch::Device& device) {
  static std::mutex mutex;
  static std::map<int, SmaxPool> pools;
  const std::lock_guard lock(mutex);
  const int index = device.has_index() ? device.index() : at::cuda::current_device();
  auto& p = pools[index];
  if (!p.slots.defined()) {
    p.slots = torch::zeros(
          {kSmaxSets, kernels::kNvfp4Slots},
          torch::TensorOptions(torch::Device(torch::kCUDA, index)).dtype(torch::kInt64));
    p.leases.resize(kSmaxSets);
    p.epochs.assign(kSmaxSets, 0);
  }
  for (int64_t n = 0; n < kSmaxSets; ++n, p.next = (p.next + 1) % kSmaxSets) {
    if (!p.leases[p.next].expired())
      continue;
    auto lease = std::make_shared<int>();
    p.leases[p.next] = lease;
    t.smax_lease = std::move(lease);
    t.smax = reinterpret_cast<unsigned long long*>(p.slots[p.next].data_ptr<int64_t>());
    t.epoch = ++p.epochs[p.next];
    p.next = (p.next + 1) % kSmaxSets;
    return;
  }
  TORCH_CHECK(false, "NVFP4: over ", kSmaxSets, " targets alive");
}

} // namespace

Nvfp4Target empty_nvfp4(
      int64_t R, int64_t C, const torch::TensorOptions& options, bool rht, bool stochastic, uint64_t seed, bool eden,
      uint64_t eden_signs, bool four_six) {
  TORCH_CHECK(R % 128 == 0 && C % 64 == 0, "empty_nvfp4: R % 128, C % 64");
  TORCH_CHECK(!eden || (!rht && !stochastic), "empty_nvfp4: MS-EDEN has its own rounding");
  TORCH_CHECK(!four_six || (!rht && !stochastic && !eden), "empty_nvfp4: 4/6 rounds to nearest");
  Nvfp4Target t{
        .data = torch::empty({R, C / 2}, options.dtype(torch::kUInt8)),
        .scale16 = torch::empty({R * C / 16}, options.dtype(torch::kInt16)),
        .rht = rht,
        .stochastic = stochastic,
        .seed = seed,
        .eden = eden,
        .eden_signs = eden_signs,
        .four_six = four_six};
  lease_smax(t, options.device());
  return t;
}

kernels::Nvfp4Out nvfp4_out(const Nvfp4Target& t, int64_t row, int64_t col) {
  TORCH_CHECK(row % 128 == 0 && col % 64 == 0, "nvfp4_out: row % 128, col % 64");
  const int64_t ld = t.data.size(1) * 2, i = row * ld + col;
  TORCH_CHECK(t.data.numel() * 2 < (int64_t{1} << 31), "nvfp4_out: tensors below 2^31 values");
  return {
        .data = static_cast<uint8_t*>(t.data.data_ptr()) + i / 2,
        .scale16 = static_cast<int16_t*>(t.scale16.data_ptr()) + ((row / 128) * (ld / 64) + col / 64) * 512,
        .smax = t.smax,
        .seed = t.seed,
        .eden_signs = t.eden_signs,
        .ld = static_cast<int32_t>(ld),
        .index0 = static_cast<int32_t>(i),
        .rht_signs = nvfp4_hadamard_signs(),
        .epoch = t.epoch,
        .rht = t.rht,
        .stochastic = t.stochastic,
        .eden = t.eden,
        .four_six = t.four_six};
}

namespace {

// t's output tensors and the kernel's view of both
std::pair<Nvfp4Tensor, kernels::Nvfp4Finish> finish_args(const Nvfp4Target& t) {
  const int64_t R = t.data.size(0), C = t.data.size(1) * 2;
  Nvfp4Tensor out{
        t.data, torch::empty({R * C / 16}, t.data.options()),
        torch::empty({}, t.data.options().dtype(torch::kFloat32))};
  const kernels::Nvfp4Finish args{t.scale16.data_ptr(),      t.smax, t.epoch, R, C, out.scale.data_ptr(),
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
  nvfp4_gemm_into(a, b, out, accumulate, alpha);
}

void quantize_nvfp4_2d(const torch::Tensor& w, const kernels::Nvfp4Out& out, const kernels::Nvfp4Out& out_t) {
  TORCH_CHECK(
        w.is_cuda() && w.dim() == 2 && w.is_contiguous() && w.size(0) % 16 == 0 && w.size(1) % 16 == 0 &&
              (w.scalar_type() == torch::kFloat32 || w.scalar_type() == torch::kBFloat16),
        "quantize_nvfp4_2d: expected contiguous 2D fp32 or bf16, dims % 16");
  kernels::quantize_nvfp4_2d(
        w.data_ptr(), w.scalar_type() == torch::kFloat32, w.size(0), w.size(1), out, out_t, stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor nvfp4_weight_values(const torch::Tensor& w_in) {
  const auto w = w_in.contiguous();
  const auto t = empty_nvfp4(w.size(0), w.size(1), w.options(), false, false, 0, false, 0, nvfp4_four_six());
  quantize_nvfp4_2d(w, nvfp4_out(t, 0, 0), {});
  return nvfp4_to_bf16(nvfp4_finish(t));
}

torch::Tensor nvfp4_gemm(const Nvfp4Tensor& a, const Nvfp4Tensor& b) {
  const int64_t M = a.data.size(0), N = b.data.size(0), K = a.data.size(1) * 2;
  TORCH_CHECK(b.data.size(1) * 2 == K, "nvfp4_gemm: shape mismatch");
  auto out = torch::empty({M, N}, a.data.options().dtype(torch::kBFloat16));
  nvfp4_gemm_into(a, b, out, false);
  return out;
}

bool nvfp4_eden_dgrad() {
  const auto* o = nvfp4_backward();
  return o != nullptr && o->dgrad && o->eden_dgrad;
}

uint64_t nvfp4_eden_signs() {
  const auto* o = nvfp4_backward();
  TORCH_CHECK(o != nullptr && o->eden_dgrad, "nvfp4_eden_signs: needs eden_dgrad on");
  // splitmix64 of a fresh seed
  uint64_t z = next_seed(*o) + 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

void set_nvfp4_four_six(bool on) {
  four_six_fwd = on;
}

bool nvfp4_four_six() {
  return four_six_fwd;
}

void set_nvfp4_backward(const Nvfp4Backward* options) {
  backward_options = options;
}

const Nvfp4Backward* nvfp4_backward() {
  return backward_options;
}

Nvfp4Target nvfp4_target(
      int64_t R, int64_t C, const torch::TensorOptions& options, Nvfp4Role role, uint64_t eden_signs) {
  if (role == Nvfp4Role::FwdInput)
    return empty_nvfp4(R, C, options, false, false, 0, false, 0, nvfp4_four_six());
  const auto* o = nvfp4_backward();
  if (o == nullptr || role == Nvfp4Role::None)
    return {};
  const bool wgrad = role == Nvfp4Role::WgradInput || role == Nvfp4Role::WgradGrad;
  if (!(wgrad ? o->wgrad : o->dgrad))
    return {};
  if (!wgrad && o->eden_dgrad)
    return empty_nvfp4(R, C, options, false, false, next_seed(*o), true, eden_signs);
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
