// Rotary embedding + RMSNorm row math shared by the CUDA kernels (included by .cu files only). One warp per row of
// head_dim values; lane pairs at columns j and half + j, j = 2 * lane + 64 * i.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>

namespace nanochat::rotary {

constexpr int kMaxIters = 4; // column pairs per lane: head_dim / 2 <= 32 lanes * 2 * kMaxIters

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__device__ __forceinline__ float2 load2(const __nv_bfloat16* p) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

__device__ __forceinline__ void store2(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

// Rotated row values held by this lane.
struct Rotated {
  float y1[kMaxIters][2], y2[kMaxIters][2];
};

// y1 = x1 * cos + x2 * sin, y2 = -x1 * sin + x2 * cos (gpt.py rotates by -theta). Returns this lane's sum of squares.
__device__ __forceinline__ float rotate(
      const __nv_bfloat16* x, const __nv_bfloat16* c, const __nv_bfloat16* s, int half, Rotated& y) {
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
__device__ __forceinline__ const __nv_bfloat16* x_row(
      const __nv_bfloat16* x, int64_t row, int heads, int head_dim, int64_t x_stride) {
  return x + row / heads * x_stride + row % heads * head_dim;
}

// One row's backward: n = y * r, dn = scale * dout: dy = r * (dn - n * mean(dn * n)); then the transposed rotation.
// emit(j, a, b): the gradient at columns j, j + 1 (unrounded).
template <typename Emit>
__device__ __forceinline__ void bwd_row(
      const __nv_bfloat16* g, const __nv_bfloat16* xr, const __nv_bfloat16* c, const __nv_bfloat16* s, float r,
      int head_dim, float scale, Emit emit) {
  const int lane = static_cast<int>(threadIdx.x % 32), half = head_dim / 2;
  Rotated y;
  rotate(xr, c, s, half, y);
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
    emit(j, dy1[0] * cj.x - dy2[0] * sj.x, dy1[1] * cj.y - dy2[1] * sj.y);
    emit(half + j, dy1[0] * sj.x + dy2[0] * cj.x, dy1[1] * sj.y + dy2[1] * cj.y);
  }
}

} // namespace nanochat::rotary
