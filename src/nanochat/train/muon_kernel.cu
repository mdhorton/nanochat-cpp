#include "nanochat/train/muon_kernel.h"

#include <algorithm>
#include <cmath>

#include <cuda_bf16.h>

namespace nanochat {

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256, kWarps = kThreads / 32, kMatThreads = 1024;
constexpr int64_t kColSegments = 16; // row segments of the column sums (wide matrices)

// torch's lerp (Lerp.h), as adamw_kernel.cu's
__device__ __forceinline__ float lerp(float self, float end, float w) {
  return fabsf(w) < 0.5f ? self + w * (end - self) : end - (end - self) * (1.f - w);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o /= 2)
    v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// the block's sum, on every thread (blockDim.x a multiple of 32)
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float part[32];
  __shared__ float total;
  v = warp_sum(v);
  if (threadIdx.x % 32 == 0)
    part[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    float s = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0.f;
    s = warp_sum(s);
    if (threadIdx.x == 0)
      total = s;
  }
  __syncthreads();
  const float t = total;
  __syncthreads(); // before part / total are reused
  return t;
}

// a warp per row: momentum, bf16 x, the row's sum of squares
__global__ void __launch_bounds__(kThreads) muon_pre_rows(
      const float* __restrict__ grad, float* __restrict__ mbuf, bf16* __restrict__ x, float* __restrict__ row_sq,
      int64_t rows, int64_t n, float momentum) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int lane = static_cast<int>(threadIdx.x % 32);
  const float w = 1.f - momentum;
  float s = 0.f;
  for (int64_t j = lane; j < n; j += 32) {
    const int64_t i = row * n + j;
    const float g = grad[i];
    const float mb = lerp(mbuf[i], g, w);
    mbuf[i] = mb;
    const bf16 q = __float2bfloat16(lerp(g, mb, momentum));
    x[i] = q;
    const float f = __bfloat162float(q);
    s = fmaf(f, f, s);
  }
  s = warp_sum(s);
  if (lane == 0)
    row_sq[row] = s;
}

// a block per matrix: row_sq -> each row's scale (in place): target / row_norm / (frobenius after * 1.01 + 1e-6)
__global__ void __launch_bounds__(kMatThreads) muon_pre_scale(float* __restrict__ row_sq, int64_t m) {
  float* sq = row_sq + blockIdx.x * m;
  float t = 0.f;
  for (int64_t r = threadIdx.x; r < m; r += blockDim.x)
    t += sq[r];
  const float target = sqrtf(block_sum(t)) / sqrtf(static_cast<float>(m));
  float eq = 0.f;
  for (int64_t r = threadIdx.x; r < m; r += blockDim.x) {
    const float s = target / fmaxf(sqrtf(sq[r]), 1e-6f);
    eq += s * s * sq[r];
  }
  const float norm = 1.f / (sqrtf(block_sum(eq)) * 1.01f + 1e-6f);
  for (int64_t r = threadIdx.x; r < m; r += blockDim.x)
    sq[r] = target / fmaxf(sqrtf(sq[r]), 1e-6f) * norm;
}

// a warp per row: x *= the row's scale
__global__ void __launch_bounds__(kThreads)
      muon_scale_rows(bf16* __restrict__ x, const float* __restrict__ scale, int64_t rows, int64_t n) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const float s = scale[row];
  for (int64_t j = threadIdx.x % 32; j < n; j += 32)
    x[row * n + j] = __float2bfloat16(__bfloat162float(x[row * n + j]) * s);
}

// a warp per row: the row's sum of squares
__global__ void __launch_bounds__(kThreads)
      muon_row_sq(const bf16* __restrict__ x, float* __restrict__ sq, int64_t rows, int64_t n) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int lane = static_cast<int>(threadIdx.x % 32);
  float s = 0.f;
  for (int64_t j = lane; j < n; j += 32) {
    const float f = __bfloat162float(x[row * n + j]);
    s = fmaf(f, f, s);
  }
  s = warp_sum(s);
  if (lane == 0)
    sq[row] = s;
}

// a thread per column and row segment (grid: columns / kThreads, kColSegments, k): partial[k][segment][n]
__global__ void __launch_bounds__(kThreads)
      muon_col_sq(const bf16* __restrict__ x, float* __restrict__ partial, int64_t m, int64_t n) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (c >= n)
    return;
  const int64_t mat = blockIdx.z, seg = blockIdx.y, per = (m + kColSegments - 1) / kColSegments;
  const bf16* xm = x + mat * m * n;
  float s = 0.f;
  const int64_t end = (seg + 1) * per < m ? (seg + 1) * per : m;
  for (int64_t r = seg * per; r < end; ++r) {
    const float f = __bfloat162float(xm[r * n + c]);
    s = fmaf(f, f, s);
  }
  partial[(mat * kColSegments + seg) * n + c] = s;
}

// a block per matrix: Muon+ and NorMuon's scale per row (rows) or column, smb updated. sq: rows: (k, len); else
// (k, kColSegments, len) partials. scale: (k, len) out.
__global__ void __launch_bounds__(kMatThreads) muon_post_scale(
      const float* __restrict__ sq_in, float* __restrict__ smb, float* __restrict__ scale, int64_t len, int64_t red,
      bool rows, float target_norm, float beta2) {
  const int64_t mat = blockIdx.x;
  const auto sq = [&](int64_t i) {
    if (rows)
      return sq_in[mat * len + i];
    float s = 0.f;
    for (int64_t seg = 0; seg < kColSegments; ++seg)
      s += sq_in[(mat * kColSegments + seg) * len + i];
    return s;
  };
  float t = 0.f;
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
    t += sq(i);
  const float frob2 = block_sum(t);
  const float cn = target_norm / fmaxf(sqrtf(frob2), 1e-6f);
  const float v_norm = cn * sqrtf(frob2); // sqrt(sum of g²) after Muon+
  const float r = static_cast<float>(red);
  float* sm = smb + mat * len;
  float acc = 0.f;
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
    const float v_mean = cn * cn * sq(i) / r;
    const float s = lerp(sm[i], v_mean, 1.f - beta2);
    sm[i] = s;
    const float step = rsqrtf(fmaxf(s, 1e-10f));
    acc += v_mean * r * step * step;
  }
  const float fs = v_norm / fmaxf(sqrtf(block_sum(acc)), 1e-10f);
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
    scale[mat * len + i] = cn * rsqrtf(fmaxf(sm[i], 1e-10f)) * fs;
}

// a warp per row: the cautious update, g = x * the row's / column's scale
__global__ void __launch_bounds__(kThreads) muon_apply(
      const bf16* __restrict__ x, float* __restrict__ p, const float* __restrict__ scale, int64_t rows, int64_t m,
      int64_t n, bool by_row, float lr, float lr_wd) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const float row_scale = by_row ? scale[row] : 0.f;
  const float* col_scale = scale + row / m * n;
  for (int64_t j = threadIdx.x % 32; j < n; j += 32) {
    const int64_t i = row * n + j;
    const float g = __bfloat162float(x[i]) * (by_row ? row_scale : col_scale[j]), pv = p[i];
    const float decay = g * pv >= 0.f ? lr_wd * pv : 0.f;
    p[i] = pv - (lr * g + decay);
  }
}

unsigned row_blocks(int64_t rows) {
  return static_cast<unsigned>((rows + kWarps - 1) / kWarps);
}

} // namespace

} // namespace nanochat

namespace nanochat::kernels {

void muon_pre(
      const float* grad, float* mbuf, void* x, float* row_sq, int64_t k, int64_t m, int64_t n, float momentum,
      cudaStream_t stream) {
  if (k * m * n == 0)
    return;
  auto* xb = static_cast<bf16*>(x);
  muon_pre_rows<<<row_blocks(k * m), kThreads, 0, stream>>>(grad, mbuf, xb, row_sq, k * m, n, momentum);
  muon_pre_scale<<<static_cast<unsigned>(k), kMatThreads, 0, stream>>>(row_sq, m);
  muon_scale_rows<<<row_blocks(k * m), kThreads, 0, stream>>>(xb, row_sq, k * m, n);
}

int64_t muon_post_scratch(int64_t k, int64_t m, int64_t n) {
  // sums (rows: k * m; columns: k * kColSegments * n) and the scales (k * len)
  return m >= n ? 2 * k * m : k * kColSegments * n + k * n;
}

void muon_post(
      const void* x, float* p, float* smb, float* scratch, int64_t k, int64_t m, int64_t n, float lr, float wd,
      float beta2, cudaStream_t stream) {
  if (k * m * n == 0)
    return;
  const auto* xb = static_cast<const bf16*>(x);
  const bool rows = m >= n;
  const int64_t len = rows ? m : n, red = rows ? n : m;
  float* sums = scratch;
  float* scale = scratch + (rows ? k * m : k * kColSegments * n);
  if (rows)
    muon_row_sq<<<row_blocks(k * m), kThreads, 0, stream>>>(xb, sums, k * m, n);
  else
    muon_col_sq<<<
          dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads), kColSegments, static_cast<unsigned>(k)), kThreads,
          0, stream>>>(xb, sums, m, n);
  const float target_norm = std::sqrt(static_cast<float>(std::min(m, n)));
  muon_post_scale<<<static_cast<unsigned>(k), kMatThreads, 0, stream>>>(
        sums, smb, scale, len, red, rows, target_norm, beta2);
  muon_apply<<<row_blocks(k * m), kThreads, 0, stream>>>(xb, p, scale, k * m, m, n, rows, lr, lr * wd);
}

} // namespace nanochat::kernels
