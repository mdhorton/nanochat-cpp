#include "nanochat/model/relu_square_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>

namespace nanochat {

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8; // bf16 per 16-byte load

using bf16 = __nv_bfloat16;

struct Vec8 {
  float v[kVec];
};

__device__ Vec8 load8(const bf16* p) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const auto* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
  Vec8 r;
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k) {
    const float2 f = __bfloat1622float2(b[k]);
    r.v[2 * k] = f.x, r.v[2 * k + 1] = f.y;
  }
  return r;
}

__device__ void store8(bf16* p, const Vec8& r) {
  uint4 raw;
  auto* b = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k)
    b[k] = __floats2bfloat162_rn(r.v[2 * k], r.v[2 * k + 1]);
  *reinterpret_cast<uint4*>(p) = raw;
}

// relu passes NaN, as torch's; bf16 * bf16 is exact in float, so the store is the op path's one rounding
__device__ float relu_square(float h) {
  const float r = h <= 0.f ? 0.f : h;
  return r * r;
}

// pow then relu backward. The factor 2 is exact, so every op order rounds g * 2h the same.
__device__ float relu_square_grad(float g, float h) {
  return __bfloat162float(__float2bfloat16(h <= 0.f ? 0.f : g * (2.f * h)));
}

__device__ __forceinline__ void relu_square_fwd_body(const bf16* h, bf16* a, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads, n_vec = n / kVec;
  for (int64_t i = blockIdx.x * kThreads + threadIdx.x; i < n_vec; i += stride) {
    Vec8 v = load8(h + i * kVec);
#pragma unroll
    for (int k = 0; k < kVec; ++k)
      v.v[k] = relu_square(v.v[k]);
    store8(a + i * kVec, v);
  }
  for (int64_t i = n_vec * kVec + blockIdx.x * kThreads + threadIdx.x; i < n; i += stride)
    a[i] = __float2bfloat16(relu_square(__bfloat162float(h[i])));
}

__device__ __forceinline__ void relu_square_bwd_body(const bf16* g, const bf16* h, bf16* dh, float* amax, int64_t n) {
  float m = 0.f;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads, n_vec = n / kVec;
  for (int64_t i = blockIdx.x * kThreads + threadIdx.x; i < n_vec; i += stride) {
    const Vec8 gv = load8(g + i * kVec), hv = load8(h + i * kVec);
    Vec8 d;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      d.v[k] = relu_square_grad(gv.v[k], hv.v[k]);
      m = fmaxf(m, fabsf(d.v[k]));
    }
    store8(dh + i * kVec, d);
  }
  for (int64_t i = n_vec * kVec + blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) {
    const float d = relu_square_grad(__bfloat162float(g[i]), __bfloat162float(h[i]));
    dh[i] = __float2bfloat16(d);
    m = fmaxf(m, fabsf(d));
  }
  if (amax == nullptr)
    return;
  // as fp8_kernel.cu's amax: block max, then one atomic on the float bits (|dh| >= 0)
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  __shared__ float smem[kThreads / 32];
  if (threadIdx.x % 32 == 0)
    smem[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < kThreads / 32; ++w)
      m = fmaxf(m, smem[w]);
    atomicMax(reinterpret_cast<int*>(amax), __float_as_int(m));
  }
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kThreads)
      nanochat_relu_square_fwd(const __nv_bfloat16* h, __nv_bfloat16* a, int64_t n) {
  nanochat::relu_square_fwd_body(h, a, n);
}

__global__ void __launch_bounds__(nanochat::kThreads) nanochat_relu_square_bwd(
      const __nv_bfloat16* g, const __nv_bfloat16* h, __nv_bfloat16* dh, float* amax, int64_t n) {
  nanochat::relu_square_bwd_body(g, h, dh, amax, n);
}

namespace nanochat::kernels {

namespace {

int blocks_for(int64_t n) {
  static const int sms = [] {
    int device = 0, s = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, device);
    return s;
  }();
  const int64_t needed = (n / kVec + kThreads - 1) / kThreads;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sms)));
}

} // namespace

void relu_square_fwd(const void* h, void* a, int64_t n, cudaStream_t stream) {
  nanochat_relu_square_fwd<<<blocks_for(n), kThreads, 0, stream>>>(
        static_cast<const bf16*>(h), static_cast<bf16*>(a), n);
}

void relu_square_bwd(const void* g, const void* h, void* dh, float* amax, int64_t n, cudaStream_t stream) {
  if (amax != nullptr)
    cudaMemsetAsync(amax, 0, sizeof(float), stream);
  nanochat_relu_square_bwd<<<blocks_for(n), kThreads, 0, stream>>>(
        static_cast<const bf16*>(g), static_cast<const bf16*>(h), static_cast<bf16*>(dh), amax, n);
}

} // namespace nanochat::kernels
