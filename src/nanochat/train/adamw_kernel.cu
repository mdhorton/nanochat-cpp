#include "nanochat/train/adamw_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>

namespace nanochat {

namespace {

constexpr int kThreads = 256;

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float load(const float* p, int64_t i) {
  return p[i];
}

__device__ __forceinline__ float load(const bf16* p, int64_t i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ void store(float* p, int64_t i, float v) {
  p[i] = v;
}

__device__ __forceinline__ void store(bf16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v); // round to nearest even, as torch's cast
}

// torch's lerp (Lerp.h): the small-weight form keeps the result near self
__device__ __forceinline__ float lerp(float self, float end, float w) {
  return fabsf(w) < 0.5f ? self + w * (end - self) : end - (end - self) * (1.f - w);
}

} // namespace

} // namespace nanochat

template <typename P, typename G>
__global__ void __launch_bounds__(nanochat::kThreads) nanochat_adamw_step(
      P* __restrict__ p, const G* __restrict__ grad, P* __restrict__ exp_avg, P* __restrict__ exp_avg_sq, int64_t n,
      nanochat::kernels::AdamWScalars s) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * nanochat::kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * nanochat::kThreads + threadIdx.x; i < n; i += stride) {
    // one torch op per line: the _rn intrinsics keep nvcc from fusing across the ops' rounding points
    const float g = nanochat::load(grad, i);
    float x = __fmul_rn(nanochat::load(p, i), s.decay); // decoupled weight decay
    const float m = nanochat::lerp(nanochat::load(exp_avg, i), g, s.w1);
    const float v = nanochat::lerp(nanochat::load(exp_avg_sq, i), __fmul_rn(g, g), s.w2);
    const float denom = __fadd_rn(sqrtf(__fmul_rn(v, s.inv_bias2)), s.eps);
    x = x + s.neg_step_size * __fdiv_rn(m, denom); // add_(exp_avg / denom, alpha = -step_size)
    nanochat::store(p, i, x);
    nanochat::store(exp_avg, i, m);
    nanochat::store(exp_avg_sq, i, v);
  }
}

namespace nanochat::kernels {

void adamw_step(
      void* p, const void* grad, void* exp_avg, void* exp_avg_sq, int64_t n, bool bf16_param, bool bf16_grad,
      const AdamWScalars& s, cudaStream_t stream) {
  if (n == 0)
    return;
  const int blocks = static_cast<int>(std::min<int64_t>((n + kThreads - 1) / kThreads, 4096));
  const auto launch = [&](auto* p_, const auto* g_) {
    using P = std::remove_pointer_t<decltype(p_)>;
    nanochat_adamw_step<P>
          <<<blocks, kThreads, 0, stream>>>(p_, g_, static_cast<P*>(exp_avg), static_cast<P*>(exp_avg_sq), n, s);
  };
  if (bf16_param)
    if (bf16_grad)
      launch(static_cast<bf16*>(p), static_cast<const bf16*>(grad));
    else
      launch(static_cast<bf16*>(p), static_cast<const float*>(grad));
  else if (bf16_grad)
    launch(static_cast<float*>(p), static_cast<const bf16*>(grad));
  else
    launch(static_cast<float*>(p), static_cast<const float*>(grad));
}

} // namespace nanochat::kernels
