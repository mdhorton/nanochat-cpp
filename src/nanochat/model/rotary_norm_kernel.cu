#include "nanochat/model/rotary_norm_kernel.h"

#include <cuda_bf16.h>

namespace nanochat {

namespace {

constexpr int kWarps = 8;    // rows per block, one warp each
constexpr int kMaxIters = 4; // column pairs per lane: head_dim / 2 <= 32 lanes * 2 * kMaxIters

__device__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__device__ float2 load2(const __nv_bfloat16* p) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

__device__ void store2(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

// Rotated row values held by this lane: pairs at columns j and half + j, j = 2 * lane + 64 * i.
struct Rotated {
  float y1[kMaxIters][2], y2[kMaxIters][2];
};

// y1 = x1 * cos + x2 * sin, y2 = -x1 * sin + x2 * cos (gpt.py rotates by -theta). Returns this lane's sum of squares.
__device__ float rotate(const __nv_bfloat16* x, const __nv_bfloat16* c, const __nv_bfloat16* s, int half, Rotated& y) {
  const int lane = static_cast<int>(threadIdx.x % 32);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < kMaxIters; ++i) {
    const int j = 2 * lane + 64 * i;
    if (j >= half)
      break;
    const float2 a = load2(x + j), b = load2(x + half + j), cj = load2(c + j), sj = load2(s + j);
    y.y1[i][0] = a.x * cj.x + b.x * sj.x;
    y.y1[i][1] = a.y * cj.y + b.y * sj.y;
    y.y2[i][0] = b.x * cj.x - a.x * sj.x;
    y.y2[i][1] = b.y * cj.y - a.y * sj.y;
    ss += y.y1[i][0] * y.y1[i][0] + y.y1[i][1] * y.y1[i][1] + y.y2[i][0] * y.y2[i][0] + y.y2[i][1] * y.y2[i][1];
  }
  return ss;
}

// x's row: token row / heads at x_stride, head row % heads within it
__device__ const __nv_bfloat16* x_row(const __nv_bfloat16* x, int64_t row, int heads, int head_dim, int64_t x_stride) {
  return x + row / heads * x_stride + row % heads * head_dim;
}

__device__ __forceinline__ void rotary_norm_fwd_body(
      const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin, __nv_bfloat16* out, float* rstd,
      int64_t rows, int heads, int64_t seq_len, int head_dim, int64_t x_stride, float scale, float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int lane = static_cast<int>(threadIdx.x % 32), half = head_dim / 2;
  const int64_t t = row / heads % seq_len;
  Rotated y;
  const float ss = warp_sum(rotate(x_row(x, row, heads, head_dim, x_stride), cos + t * half, sin + t * half, half, y));
  const float r = rsqrtf(ss / static_cast<float>(head_dim) + eps);
  if (lane == 0)
    rstd[row] = r;
  auto* o = out + row * head_dim;
  const float k = r * scale;
#pragma unroll
  for (int i = 0; i < kMaxIters; ++i) {
    const int j = 2 * lane + 64 * i;
    if (j >= half)
      break;
    store2(o + j, y.y1[i][0] * k, y.y1[i][1] * k);
    store2(o + half + j, y.y2[i][0] * k, y.y2[i][1] * k);
  }
}

// n = y * r, dn = scale * dout: dy = r * (dn - n * mean(dn * n)); then the transposed rotation.
__device__ __forceinline__ void rotary_norm_bwd_body(
      const __nv_bfloat16* dout, const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
      const float* rstd, __nv_bfloat16* dx, int64_t rows, int heads, int64_t seq_len, int head_dim, int64_t x_stride,
      float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int lane = static_cast<int>(threadIdx.x % 32), half = head_dim / 2;
  const int64_t t = row / heads % seq_len;
  const auto *c = cos + t * half, *s = sin + t * half, *g = dout + row * head_dim;
  Rotated y;
  rotate(x_row(x, row, heads, head_dim, x_stride), c, s, half, y);
  const float r = rstd[row];
  float dn1[kMaxIters][2], dn2[kMaxIters][2], dot = 0.f;
#pragma unroll
  for (int i = 0; i < kMaxIters; ++i) {
    const int j = 2 * lane + 64 * i;
    if (j >= half)
      break;
    const float2 g1 = load2(g + j), g2 = load2(g + half + j);
    dn1[i][0] = g1.x * scale, dn1[i][1] = g1.y * scale, dn2[i][0] = g2.x * scale, dn2[i][1] = g2.y * scale;
    dot += dn1[i][0] * y.y1[i][0] + dn1[i][1] * y.y1[i][1] + dn2[i][0] * y.y2[i][0] + dn2[i][1] * y.y2[i][1];
  }
  const float mean = warp_sum(dot) * r / static_cast<float>(head_dim); // mean(dn * n), n = y * r
  auto* d = dx + row * head_dim;
#pragma unroll
  for (int i = 0; i < kMaxIters; ++i) {
    const int j = 2 * lane + 64 * i;
    if (j >= half)
      break;
    const float2 cj = load2(c + j), sj = load2(s + j);
    float dy1[2], dy2[2];
    for (int e = 0; e < 2; ++e) {
      dy1[e] = r * (dn1[i][e] - y.y1[i][e] * r * mean);
      dy2[e] = r * (dn2[i][e] - y.y2[i][e] * r * mean);
    }
    store2(d + j, dy1[0] * cj.x - dy2[0] * sj.x, dy1[1] * cj.y - dy2[1] * sj.y);
    store2(d + half + j, dy1[0] * sj.x + dy2[0] * cj.x, dy1[1] * sj.y + dy2[1] * cj.y);
  }
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kWarps * 32) nanochat_rotary_norm_fwd(
      const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin, __nv_bfloat16* out, float* rstd,
      int64_t rows, int heads, int64_t seq_len, int head_dim, int64_t x_stride, float scale, float eps) {
  nanochat::rotary_norm_fwd_body(x, cos, sin, out, rstd, rows, heads, seq_len, head_dim, x_stride, scale, eps);
}

__global__ void __launch_bounds__(nanochat::kWarps * 32) nanochat_rotary_norm_bwd(
      const __nv_bfloat16* dout, const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
      const float* rstd, __nv_bfloat16* dx, int64_t rows, int heads, int64_t seq_len, int head_dim, int64_t x_stride,
      float scale) {
  nanochat::rotary_norm_bwd_body(dout, x, cos, sin, rstd, dx, rows, heads, seq_len, head_dim, x_stride, scale);
}

namespace nanochat::kernels {

namespace {

using bf16 = __nv_bfloat16;

unsigned blocks(int64_t rows) {
  return static_cast<unsigned>((rows + kWarps - 1) / kWarps);
}

} // namespace

void rotary_norm_fwd(
      const void* x, const void* cos, const void* sin, void* out, float* rstd, int64_t rows, int heads, int64_t seq_len,
      int head_dim, int64_t x_stride, float scale, float eps, cudaStream_t stream) {
  nanochat_rotary_norm_fwd<<<blocks(rows), kWarps * 32, 0, stream>>>(
        static_cast<const bf16*>(x), static_cast<const bf16*>(cos), static_cast<const bf16*>(sin),
        static_cast<bf16*>(out), rstd, rows, heads, seq_len, head_dim, x_stride, scale, eps);
}

void rotary_norm_bwd(
      const void* dout, const void* x, const void* cos, const void* sin, const float* rstd, void* dx, int64_t rows,
      int heads, int64_t seq_len, int head_dim, int64_t x_stride, float scale, cudaStream_t stream) {
  nanochat_rotary_norm_bwd<<<blocks(rows), kWarps * 32, 0, stream>>>(
        static_cast<const bf16*>(dout), static_cast<const bf16*>(x), static_cast<const bf16*>(cos),
        static_cast<const bf16*>(sin), rstd, static_cast<bf16*>(dx), rows, heads, seq_len, head_dim, x_stride, scale);
}

} // namespace nanochat::kernels
