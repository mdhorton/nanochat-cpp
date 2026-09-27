#include "nanochat/model/residual_norm_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>

namespace nanochat {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32; // rows per block pass, one warp each
constexpr int kVec = 8;               // bf16 per 16-byte load

using bf16 = __nv_bfloat16;

struct Vec8 {
  float v[kVec];
};

__device__ Vec8 unpack8(uint4 raw) {
  const auto* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
  Vec8 r;
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k) {
    const float2 f = __bfloat1622float2(b[k]);
    r.v[2 * k] = f.x, r.v[2 * k + 1] = f.y;
  }
  return r;
}

__device__ uint4 pack8(const Vec8& r) {
  uint4 raw;
  auto* b = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k)
    b[k] = __floats2bfloat162_rn(r.v[2 * k], r.v[2 * k + 1]);
  return raw;
}

__device__ uint4 load_raw(const void* p, int64_t i) {
  return static_cast<const uint4*>(p)[i / kVec];
}

__device__ Vec8 load8(const void* p, int64_t i) {
  return unpack8(load_raw(p, i));
}

__device__ void store8(void* p, int64_t i, const Vec8& v) {
  static_cast<uint4*>(p)[i / kVec] = pack8(v);
}

__device__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__device__ float block_sum(float v, float* smem) {
  v = warp_sum(v);
  __syncthreads(); // smem reuse across calls
  if (threadIdx.x % 32 == 0)
    smem[threadIdx.x / 32] = v;
  __syncthreads();
  v = 0.f;
  if (threadIdx.x == 0)
    for (int w = 0; w < kWarps; ++w)
      v += smem[w];
  return v; // valid in thread 0
}

// Op path roundings: x + r is one bf16 op; the blend rounds the fp32 lambdas to bf16 (type promotion) and each
// product, bf16(bf16(lr * s) + bf16(l0 * x0)). Full-precision blend math trained measurably worse (d12 300 steps:
// val bpb 1.052 vs 1.028). The norm is fp32 inside, rounded once.
template <int kIters>
__device__ __forceinline__ void residual_norm_fwd_body(const kernels::ResidualNormFwd& a) {
  const int lane = static_cast<int>(threadIdx.x % 32);
  const bool blend = a.x0 != nullptr;
  const float lr = blend ? round_bf16(*a.lr) : 0.f, l0 = blend ? round_bf16(*a.l0) : 0.f;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32; row < a.rows;
       row += static_cast<int64_t>(gridDim.x) * kWarps) {
    uint4 res[kIters];
    float ss = 0.f;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int col = (it * 32 + lane) * kVec;
      if (col >= a.cols)
        break;
      const int64_t i = row * a.cols + col;
      Vec8 v = load8(a.x, i);
      if (a.r != nullptr) {
        const Vec8 r = load8(a.r, i);
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          v.v[k] = round_bf16(v.v[k] + r.v[k]);
        if (a.s != nullptr)
          store8(a.s, i, v);
      }
      if (blend) {
        const Vec8 x0 = load8(a.x0, i);
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          v.v[k] = round_bf16(round_bf16(lr * v.v[k]) + round_bf16(l0 * x0.v[k]));
      }
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        ss += v.v[k] * v.v[k];
      res[it] = pack8(v);
      static_cast<uint4*>(a.res)[i / kVec] = res[it];
    }
    const float rs = rsqrtf(warp_sum(ss) / static_cast<float>(a.cols) + a.eps);
    if (lane == 0)
      a.rstd[row] = rs;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int col = (it * 32 + lane) * kVec;
      if (col >= a.cols)
        break;
      Vec8 v = unpack8(res[it]);
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        v.v[k] *= rs;
      store8(a.n, row * a.cols + col, v);
    }
  }
}

// n = res * rs: dres = rs * (g_n - res * rs^2 * mean(g_n * res)), rounded to bf16 as the norm's output gradient, then
// added to g_res in bf16 (autograd's accumulation), then the blend backward.
template <int kIters>
__device__ __forceinline__ void residual_norm_bwd_body(const kernels::ResidualNormBwd& a) {
  __shared__ float smem[kWarps];
  const int lane = static_cast<int>(threadIdx.x % 32);
  const bool blend = a.x0 != nullptr;
  const float lr = blend ? round_bf16(*a.lr) : 0.f, l0 = blend ? round_bf16(*a.l0) : 0.f;
  float sr = 0.f, s0 = 0.f;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32; row < a.rows;
       row += static_cast<int64_t>(gridDim.x) * kWarps) {
    uint4 res[kIters], gn[kIters];
    float c = 0.f, rs = 0.f;
    if (a.g_n != nullptr) {
      float dot = 0.f;
#pragma unroll
      for (int it = 0; it < kIters; ++it) {
        const int col = (it * 32 + lane) * kVec;
        if (col >= a.cols)
          break;
        const int64_t i = row * a.cols + col;
        res[it] = load_raw(a.res, i);
        gn[it] = load_raw(a.g_n, i);
        const Vec8 r = unpack8(res[it]), g = unpack8(gn[it]);
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          dot += g.v[k] * r.v[k];
      }
      rs = a.rstd[row];
      c = warp_sum(dot) * rs * rs / static_cast<float>(a.cols);
    }
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int col = (it * 32 + lane) * kVec;
      if (col >= a.cols)
        break;
      const int64_t i = row * a.cols + col;
      Vec8 d{};
      if (a.g_n != nullptr) {
        const Vec8 r = unpack8(res[it]), g = unpack8(gn[it]);
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          d.v[k] = round_bf16(rs * (g.v[k] - r.v[k] * c));
      }
      if (a.g_res != nullptr) {
        const Vec8 g = load8(a.g_res, i);
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          d.v[k] = round_bf16(d.v[k] + g.v[k]);
      }
      if (!blend) {
        store8(a.ds, i, d);
        continue;
      }
      const Vec8 s = load8(a.s, i), x0 = load8(a.x0, i);
      Vec8 ds, dx0 = a.dx0_sum ? load8(a.dx0, i) : Vec8{};
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        sr += d.v[k] * s.v[k];
        s0 += d.v[k] * x0.v[k];
        ds.v[k] = lr * d.v[k];
        if (a.dx0_add_ds)
          dx0.v[k] = round_bf16(dx0.v[k] + round_bf16(ds.v[k]));
        dx0.v[k] += round_bf16(l0 * d.v[k]);
      }
      if (a.ds != nullptr)
        store8(a.ds, i, ds);
      store8(a.dx0, i, dx0);
    }
  }
  if (!blend)
    return;
  sr = block_sum(sr, smem);
  s0 = block_sum(s0, smem);
  if (threadIdx.x == 0) {
    a.partials[2 * blockIdx.x] = sr;
    a.partials[2 * blockIdx.x + 1] = s0;
  }
}

// One block: sums the partials in a fixed order (deterministic), writes both whole gradient vectors.
__device__ __forceinline__ void residual_norm_bwd_finalize_body(
      const float* partials, int blocks, float* dlr, float* dl0, int layer, int n_layer) {
  __shared__ float smem[kWarps];
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

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu); the
// suffix is the largest cols each handles (row values stay in registers).
#define NANOCHAT_RESIDUAL_NORM(cols)                                                                                   \
  __global__ void __launch_bounds__(nanochat::kThreads)                                                                \
        nanochat_residual_norm_fwd_c##cols(const nanochat::kernels::ResidualNormFwd a) {                               \
    nanochat::residual_norm_fwd_body<(cols) / 256>(a);                                                                 \
  }                                                                                                                    \
  __global__ void __launch_bounds__(nanochat::kThreads)                                                                \
        nanochat_residual_norm_bwd_c##cols(const nanochat::kernels::ResidualNormBwd a) {                               \
    nanochat::residual_norm_bwd_body<(cols) / 256>(a);                                                                 \
  }

NANOCHAT_RESIDUAL_NORM(256)
NANOCHAT_RESIDUAL_NORM(512)
NANOCHAT_RESIDUAL_NORM(768)
NANOCHAT_RESIDUAL_NORM(1024)
NANOCHAT_RESIDUAL_NORM(1280)
NANOCHAT_RESIDUAL_NORM(1536)
NANOCHAT_RESIDUAL_NORM(1792)
NANOCHAT_RESIDUAL_NORM(2048)

__global__ void __launch_bounds__(nanochat::kThreads) nanochat_residual_norm_bwd_finalize(
      const float* partials, int blocks, float* dlr, float* dl0, int layer, int n_layer) {
  nanochat::residual_norm_bwd_finalize_body(partials, blocks, dlr, dl0, layer, n_layer);
}

namespace nanochat::kernels {

namespace {

int blocks_for(int64_t rows) {
  static const int sms = [] {
    int device = 0, n = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    return n;
  }();
  const int64_t needed = (rows + kWarps - 1) / kWarps;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sms)));
}

template <typename Args>
using Kernel = void (*)(Args);

template <typename Args>
Kernel<Args> select(int cols, const Kernel<Args> (&kernels)[kResidualNormMaxCols / 256]) {
  return kernels[(cols + 255) / 256 - 1];
}

} // namespace

void residual_norm_fwd(const ResidualNormFwd& a, cudaStream_t stream) {
  static constexpr Kernel<ResidualNormFwd> kernels[] = {
        nanochat_residual_norm_fwd_c256,  nanochat_residual_norm_fwd_c512,  nanochat_residual_norm_fwd_c768,
        nanochat_residual_norm_fwd_c1024, nanochat_residual_norm_fwd_c1280, nanochat_residual_norm_fwd_c1536,
        nanochat_residual_norm_fwd_c1792, nanochat_residual_norm_fwd_c2048};
  select(a.cols, kernels)<<<blocks_for(a.rows), kThreads, 0, stream>>>(a);
}

int residual_norm_bwd_blocks(int64_t rows) {
  return blocks_for(rows);
}

void residual_norm_bwd(const ResidualNormBwd& a, cudaStream_t stream) {
  static constexpr Kernel<ResidualNormBwd> kernels[] = {
        nanochat_residual_norm_bwd_c256,  nanochat_residual_norm_bwd_c512,  nanochat_residual_norm_bwd_c768,
        nanochat_residual_norm_bwd_c1024, nanochat_residual_norm_bwd_c1280, nanochat_residual_norm_bwd_c1536,
        nanochat_residual_norm_bwd_c1792, nanochat_residual_norm_bwd_c2048};
  const int blocks = blocks_for(a.rows);
  select(a.cols, kernels)<<<blocks, kThreads, 0, stream>>>(a);
  if (a.x0 != nullptr)
    nanochat_residual_norm_bwd_finalize<<<1, kThreads, 0, stream>>>(
          a.partials, blocks, a.dlr, a.dl0, a.layer, a.n_layer);
}

} // namespace nanochat::kernels
