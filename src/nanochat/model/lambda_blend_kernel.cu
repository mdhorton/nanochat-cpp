#include "nanochat/model/lambda_blend_kernel.h"

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

// Rounds as the op-by-op path (fp32 lambda * bf16 tensor promotes to bf16): the lambdas are rounded to bf16, and
// bf16(bf16(a * x) + bf16(b * x0)). Full-precision math shifts training measurably (d12 300 steps: val bpb 1.052 vs
// 1.028), so the kernel keeps Python's roundings.
__device__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ float blend(float a, float x, float b, float x0) {
  return round_bf16(a * x) + round_bf16(b * x0);
}

__device__ float block_sum(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __syncthreads(); // smem reuse across calls
  if (lane == 0)
    smem[warp] = v;
  __syncthreads();
  v = 0.f;
  if (threadIdx.x == 0)
    for (int w = 0; w < kThreads / 32; ++w)
      v += smem[w];
  return v; // valid in thread 0
}

__device__ __forceinline__ void lambda_blend_fwd_body(
      const bf16* x, const bf16* x0, const float* lr, const float* l0, bf16* out, int64_t n) {
  const float a = round_bf16(*lr), b = round_bf16(*l0);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads, n_vec = n / kVec;
  for (int64_t i = blockIdx.x * kThreads + threadIdx.x; i < n_vec; i += stride) {
    const Vec8 xv = load8(x + i * kVec), x0v = load8(x0 + i * kVec);
    Vec8 o;
#pragma unroll
    for (int k = 0; k < kVec; ++k)
      o.v[k] = blend(a, xv.v[k], b, x0v.v[k]);
    store8(out + i * kVec, o);
  }
  for (int64_t i = n_vec * kVec + blockIdx.x * kThreads + threadIdx.x; i < n; i += stride)
    out[i] = __float2bfloat16(blend(a, __bfloat162float(x[i]), b, __bfloat162float(x0[i])));
}

__device__ __forceinline__ void lambda_blend_bwd_body(
      const bf16* g, const bf16* x, const bf16* x0, const float* lr, const float* l0, bf16* dx, bf16* dx0,
      float* partials, int64_t n) {
  __shared__ float smem[kThreads / 32];
  const float a = round_bf16(*lr), b = round_bf16(*l0);
  float sr = 0.f, s0 = 0.f;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads, n_vec = n / kVec;
  for (int64_t i = blockIdx.x * kThreads + threadIdx.x; i < n_vec; i += stride) {
    const Vec8 gv = load8(g + i * kVec), xv = load8(x + i * kVec), x0v = load8(x0 + i * kVec);
    Vec8 d, d0;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      d.v[k] = a * gv.v[k];
      d0.v[k] = b * gv.v[k];
      sr += gv.v[k] * xv.v[k];
      s0 += gv.v[k] * x0v.v[k];
    }
    store8(dx + i * kVec, d);
    store8(dx0 + i * kVec, d0);
  }
  for (int64_t i = n_vec * kVec + blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) {
    const float gi = __bfloat162float(g[i]);
    dx[i] = __float2bfloat16(a * gi);
    dx0[i] = __float2bfloat16(b * gi);
    sr += gi * __bfloat162float(x[i]);
    s0 += gi * __bfloat162float(x0[i]);
  }
  sr = block_sum(sr, smem);
  s0 = block_sum(s0, smem);
  if (threadIdx.x == 0) {
    partials[2 * blockIdx.x] = sr;
    partials[2 * blockIdx.x + 1] = s0;
  }
}

// One block: sums the partials in a fixed order, writes both whole gradient vectors.
__device__ __forceinline__ void lambda_blend_bwd_finalize_body(
      const float* partials, int blocks, float* dlr, float* dl0, int layer, int n_layer) {
  __shared__ float smem[kThreads / 32];
  float sr = 0.f, s0 = 0.f;
  for (int i = threadIdx.x; i < blocks; i += kThreads) {
    sr += partials[2 * i];
    s0 += partials[2 * i + 1];
  }
  sr = block_sum(sr, smem);
  s0 = block_sum(s0, smem);
  for (int j = threadIdx.x; j < n_layer; j += kThreads)
    if (j != layer)
      dlr[j] = 0.f, dl0[j] = 0.f;
  if (threadIdx.x == 0) // the sums are valid in thread 0
    dlr[layer] = sr, dl0[layer] = s0;
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kThreads) nanochat_lambda_blend_fwd(
      const __nv_bfloat16* x, const __nv_bfloat16* x0, const float* lr, const float* l0, __nv_bfloat16* out,
      int64_t n) {
  nanochat::lambda_blend_fwd_body(x, x0, lr, l0, out, n);
}

__global__ void __launch_bounds__(nanochat::kThreads) nanochat_lambda_blend_bwd(
      const __nv_bfloat16* g, const __nv_bfloat16* x, const __nv_bfloat16* x0, const float* lr, const float* l0,
      __nv_bfloat16* dx, __nv_bfloat16* dx0, float* partials, int64_t n) {
  nanochat::lambda_blend_bwd_body(g, x, x0, lr, l0, dx, dx0, partials, n);
}

__global__ void __launch_bounds__(nanochat::kThreads) nanochat_lambda_blend_bwd_finalize(
      const float* partials, int blocks, float* dlr, float* dl0, int layer, int n_layer) {
  nanochat::lambda_blend_bwd_finalize_body(partials, blocks, dlr, dl0, layer, n_layer);
}

namespace nanochat::kernels {

namespace {

int sm_count() {
  static const int sms = [] {
    int device = 0, n = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    return n;
  }();
  return sms;
}

int blocks_for(int64_t n) {
  const int64_t needed = (n / kVec + kThreads - 1) / kThreads;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sm_count())));
}

} // namespace

int lambda_blend_bwd_blocks(int64_t n) {
  return blocks_for(n);
}

void lambda_blend_fwd(
      const void* x, const void* x0, const float* lr, const float* l0, void* out, int64_t n, cudaStream_t stream) {
  nanochat_lambda_blend_fwd<<<blocks_for(n), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x), static_cast<const bf16*>(x0), lr, l0, static_cast<bf16*>(out), n);
}

void lambda_blend_bwd(
      const void* g, const void* x, const void* x0, const float* lr, const float* l0, void* dx, void* dx0,
      float* partials, float* dlr, float* dl0, int layer, int n_layer, int64_t n, cudaStream_t stream) {
  const int blocks = blocks_for(n);
  nanochat_lambda_blend_bwd<<<blocks, kThreads, 0, stream>>>(
        static_cast<const bf16*>(g), static_cast<const bf16*>(x), static_cast<const bf16*>(x0), lr, l0,
        static_cast<bf16*>(dx), static_cast<bf16*>(dx0), partials, n);
  nanochat_lambda_blend_bwd_finalize<<<1, kThreads, 0, stream>>>(partials, blocks, dlr, dl0, layer, n_layer);
}

} // namespace nanochat::kernels
