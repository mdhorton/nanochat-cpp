#include "nanochat/model/ops/backout_kernel.h"

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

__device__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ float block_sum(float v) {
  __shared__ float smem[kThreads / 32];
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  if (threadIdx.x % 32 == 0)
    smem[threadIdx.x / 32] = v;
  __syncthreads();
  v = 0.f;
  if (threadIdx.x == 0)
    for (int w = 0; w < kThreads / 32; ++w)
      v += smem[w];
  return v; // valid in thread 0
}

__device__ __forceinline__ void backout_fwd_body(
      const bf16* x, const bf16* xb, const float* lambda, bf16* out, int64_t n) {
  const float lam = round_bf16(*lambda);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads * kVec;
  for (int64_t i = (static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x) * kVec; i < n; i += stride) {
    Vec8 v = load8(x + i);
    const Vec8 b = load8(xb + i);
#pragma unroll
    for (int k = 0; k < kVec; ++k)
      v.v[k] = round_bf16(v.v[k] - round_bf16(lam * b.v[k]));
    store8(out + i, v);
  }
}

__device__ __forceinline__ void backout_bwd_body(
      const bf16* g, const bf16* xb, const float* lambda, bf16* dxb, float* partials, int64_t n) {
  const float lam = round_bf16(*lambda);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads * kVec;
  float dl = 0.f;
  for (int64_t i = (static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x) * kVec; i < n; i += stride) {
    Vec8 d = load8(g + i);
    const Vec8 b = load8(xb + i);
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      dl -= d.v[k] * b.v[k];
      d.v[k] = -lam * d.v[k];
    }
    store8(dxb + i, d);
  }
  dl = block_sum(dl);
  if (threadIdx.x == 0)
    partials[blockIdx.x] = dl;
}

} // namespace

} // namespace nanochat

// Kernels: global nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kThreads) nanochat_backout_fwd(
      const __nv_bfloat16* x, const __nv_bfloat16* xb, const float* lambda, __nv_bfloat16* out, int64_t n) {
  nanochat::backout_fwd_body(x, xb, lambda, out, n);
}

__global__ void __launch_bounds__(nanochat::kThreads) nanochat_backout_bwd(
      const __nv_bfloat16* g, const __nv_bfloat16* xb, const float* lambda, __nv_bfloat16* dxb, float* partials,
      int64_t n) {
  nanochat::backout_bwd_body(g, xb, lambda, dxb, partials, n);
}

// One block: sums the partials in a fixed order (deterministic).
__global__ void __launch_bounds__(nanochat::kThreads)
      nanochat_backout_bwd_finalize(const float* partials, int blocks, float* dlambda) {
  float s = 0.f;
  for (int i = threadIdx.x; i < blocks; i += nanochat::kThreads)
    s += partials[i];
  s = nanochat::block_sum(s);
  if (threadIdx.x == 0)
    *dlambda = s;
}

namespace nanochat::kernels {

namespace {

int blocks_for(int64_t n) {
  static const int sms = [] {
    int device = 0, count = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    return count;
  }();
  const int64_t needed = (n / kVec + kThreads - 1) / kThreads;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sms)));
}

} // namespace

void backout_fwd(const void* x, const void* xb, const float* lambda, void* out, int64_t n, cudaStream_t stream) {
  nanochat_backout_fwd<<<blocks_for(n), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x), static_cast<const bf16*>(xb), lambda, static_cast<bf16*>(out), n);
}

int backout_bwd_blocks(int64_t n) {
  return blocks_for(n);
}

void backout_bwd(
      const void* g, const void* xb, const float* lambda, void* dxb, float* partials, float* dlambda, int64_t n,
      cudaStream_t stream) {
  const int blocks = blocks_for(n);
  nanochat_backout_bwd<<<blocks, kThreads, 0, stream>>>(
        static_cast<const bf16*>(g), static_cast<const bf16*>(xb), lambda, static_cast<bf16*>(dxb), partials, n);
  nanochat_backout_bwd_finalize<<<1, kThreads, 0, stream>>>(partials, blocks, dlambda);
}

} // namespace nanochat::kernels
