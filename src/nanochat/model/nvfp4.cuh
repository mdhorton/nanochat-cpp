// NVFP4 rounding shared by the simulation (nvfp4_sim_kernel.cu), the conversion from MX (nvfp4_kernel.cu) and the
// kernels that write NVFP4 directly (kernels::Nvfp4Out), so all give the same bits: e2m1 values with a ue4m3 scale per
// 16, under an fp32 tensor scale amax / (6 * 448). sm_120a (hardware e2m1 conversion).
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "nanochat/model/fp8_kernel.h"
#include "nanochat/model/scale_layout.cuh"

namespace nanochat {

constexpr float kE2m1Max = 6.f, kE4m3Max = 448.f;

// e2m1 magnitudes by code
__device__ constexpr float kE2m1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

// e2m1 codes (sign in bit 3) of 8 values, value k in bits 4k..4k+3: round to nearest, ties to even, saturating at 6
__device__ __forceinline__ uint32_t e2m1x8_rn(const float (&a)[8]) {
  uint32_t out;
  asm("{\n"
      " .reg .b8 b0, b1, b2, b3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      " mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(out)
      : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]), "f"(a[4]), "f"(a[5]), "f"(a[6]), "f"(a[7]));
  return out;
}

// a rounded stochastically onto e2m1's grid (0, .5, 1, 1.5, 2, 3, 4, 6), saturating at 6: 8 random bits (r's low
// byte) added below the kept mantissa bit, then truncated; below 1 (fixed spacing .5) as a + 1, so exact
__device__ __forceinline__ float e2m1_sr(float a, uint32_t r) {
  const float m = fminf(fabsf(a), kE2m1Max);
  const bool low = m < 1.f;
  const float t = low ? m + 1.f : m;
  const float g = __uint_as_float((__float_as_uint(t) + ((r & 0xffu) << 14)) & 0xffc00000u);
  return copysignf(low ? g - 1.f : g, a);
}

// 32 random bits from (seed, i), i < 2^32: 32-bit multiplies only (64-bit ones are slow here), murmur3's finalizer
__device__ __forceinline__ uint32_t nvfp4_random(uint64_t seed, uint64_t i) {
  uint32_t x = static_cast<uint32_t>(i) * 0x9e3779b1u + static_cast<uint32_t>(seed) * 0x85ebca77u +
               static_cast<uint32_t>(seed >> 32) * 0xc2b2ae3du;
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

// e2m1 codes (as e2m1x8_rn) of 8 values v * to_q from the index-th (% 8) value of a tensor: round to nearest, or
// stochastic, 8 random bits per value, nvfp4_random(seed, index / 4 + k) for values 4k .. 4k + 3
__device__ __forceinline__ uint32_t nvfp4_codes8(
      const float* v, float to_q, bool stochastic, uint64_t seed, int64_t index) {
  float a[8];
#pragma unroll
  for (int k = 0; k < 8; ++k)
    a[k] = v[k] * to_q;
  if (stochastic) {
    const uint32_t r[2] = {nvfp4_random(seed, index / 4), nvfp4_random(seed, index / 4 + 1)};
#pragma unroll
    for (int k = 0; k < 8; ++k)
      a[k] = e2m1_sr(a[k], r[k / 4] >> (8 * (k % 4)));
  }
  return e2m1x8_rn(a);
}

// the tensor's encode scale: amax maps to 6 * 448
__device__ __forceinline__ float nvfp4_encode(float tensor_amax) {
  return tensor_amax > 0.f ? kE2m1Max * kE4m3Max / tensor_amax : 0.f;
}

// A 16-value block's ue4m3 scale (s8) and the multipliers to e2m1 units (to_q) and back to values (from_q).
struct Nvfp4Scale {
  __nv_fp8_storage_t s8;
  float to_q, from_q;
};

__device__ __forceinline__ Nvfp4Scale nvfp4_scale(float block_amax, float enc) {
  const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(block_amax / kE2m1Max * enc, __NV_SATFINITE, __NV_E4M3);
  const float s = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3)));
  return {s8, s > 0.f ? enc / s : 0.f, s > 0.f ? s / enc : 0.f};
}

__device__ __forceinline__ float nvfp4_value(int code, float from_q) {
  const float m = kE2m1[code & 7] * from_q;
  return code & 8 ? -m : m;
}

// Kernels writing NVFP4 directly (kernels::Nvfp4Out): the tensor scale is a power of two set afterwards
// (nvfp4_finish), so a block's scale is e4m3's rounding of its amax / 6 at any exponent, which the power of two then
// shifts exactly: the same codes as nvfp4_scale's at that tensor scale, except for blocks whose scale lands among
// e4m3's subnormals (over 2^14 below the largest), whose codes ignore that coarser rounding.

// e4m3's rounding of x >= 0 without its exponent range: 3 mantissa bits, ties to even
__device__ __forceinline__ float nvfp4_round_scale(float x) {
  const uint32_t b = __float_as_uint(x);
  return __uint_as_float((b + 0x7ffffu + ((b >> 20) & 1u)) & 0xfff00000u);
}

// m / 6 without the division's slow-path branch (which splits the caller's scheduling): correctly rounded but for
// m < 2^-125 (checked exhaustively), whose blocks' scales vanish under any tensor scale anyway
__device__ __forceinline__ float nvfp4_div6(float m) {
  const float c = 1.f / kE2m1Max, q = m * c;
  return fmaf(fmaf(-q, kE2m1Max, m), c, q);
}

// 1 / s for nvfp4_round_scale's s (at most 4 significant bits) below 2^126, as __frcp_rn without its slow-path branch
// (checked exhaustively)
__device__ __forceinline__ float nvfp4_rcp(float s) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(s));
  return fmaf(fmaf(-s, y, 1.f), y, y);
}

// 8 values' codes for o; index: the first value's in the whole tensor
__device__ __forceinline__ uint32_t nvfp4_pack8(const float* v, float to_q, const kernels::Nvfp4Out& o, int64_t index) {
  return nvfp4_codes8(v, to_q, o.stochastic, o.seed, index);
}

// x . H per 16 values (nvfp4_hadamard: signs / 4, then the fast Walsh-Hadamard transform, as nvfp4_kernel.cu's)
__device__ __forceinline__ void nvfp4_rht16(float (&v)[16], uint32_t signs) {
#pragma unroll
  for (int i = 0; i < 16; ++i)
    v[i] *= (signs >> i) & 1u ? -.25f : .25f;
#pragma unroll
  for (int h = 1; h < 16; h *= 2)
#pragma unroll
    for (int i = 0; i < 16; i += 2 * h)
#pragma unroll
      for (int j = i; j < i + h; ++j) {
        const float a = v[j], b = v[j + h];
        v[j] = a + b, v[j + h] = a - b;
      }
}

// nvfp4_rht16 split over a lane pair (lane ^ 1): high holds values 8..15
__device__ __forceinline__ void nvfp4_rht8(float (&v)[8], uint32_t signs, bool high) {
#pragma unroll
  for (int i = 0; i < 8; ++i)
    v[i] *= (signs >> (i + (high ? 8 : 0))) & 1u ? -.25f : .25f;
#pragma unroll
  for (int h = 1; h < 8; h *= 2)
#pragma unroll
    for (int i = 0; i < 8; i += 2 * h)
#pragma unroll
      for (int j = i; j < i + h; ++j) {
        const float a = v[j], b = v[j + h];
        v[j] = a + b, v[j + h] = a - b;
      }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float p = __shfl_xor_sync(0xffffffff, v[i], 1);
    v[i] = high ? p - v[i] : v[i] + p;
  }
}

__device__ __forceinline__ void nvfp4_store_scale(const kernels::Nvfp4Out& o, int64_t row, int64_t col, float s) {
  static_cast<uint16_t*>(o.scale16)[mx_scale_index(row, col / 16, o.ld / 64)] = static_cast<uint16_t>(
        __float_as_uint(s) >> 16); // exact: 3 bits
}

// One thread's 16 values of row `row` from col (% 16): RHT (o.rht), block scale, codes. Returns the block scale.
__device__ __forceinline__ float nvfp4_store16(float (&v)[16], const kernels::Nvfp4Out& o, int64_t row, int64_t col) {
  if (o.rht)
    nvfp4_rht16(v, o.rht_signs);
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k)
    m = fmaxf(m, fabsf(v[k]));
  const float s = nvfp4_round_scale(nvfp4_div6(m)), to_q = s > 0.f ? nvfp4_rcp(s) : 0.f;
  const int64_t i = row * o.ld + col;
  *reinterpret_cast<uint2*>(static_cast<uint8_t*>(o.data) + i / 2) = make_uint2(
        nvfp4_pack8(v, to_q, o, o.index0 + i), nvfp4_pack8(v + 8, to_q, o, o.index0 + i + 8));
  nvfp4_store_scale(o, row, col, s);
  return s;
}

// nvfp4_store8's stores, the block's amax m given (no rht)
__device__ __forceinline__ float nvfp4_store8_amax(
      const float (&v)[8], float m, const kernels::Nvfp4Out& o, int64_t row, int64_t col) {
  const float s = nvfp4_round_scale(nvfp4_div6(m)), to_q = s > 0.f ? nvfp4_rcp(s) : 0.f;
  const int64_t i = row * o.ld + col;
  *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(o.data) + i / 2) = nvfp4_pack8(v, to_q, o, o.index0 + i);
  if (col % 16 == 0)
    nvfp4_store_scale(o, row, col, s);
  return s;
}

// nvfp4_store16 over a lane pair (lane ^ 1), 8 values each from col (% 8). Both lanes return the block scale.
__device__ __forceinline__ float nvfp4_store8(float (&v)[8], const kernels::Nvfp4Out& o, int64_t row, int64_t col) {
  if (o.rht)
    nvfp4_rht8(v, o.rht_signs, col % 16 != 0);
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k)
    m = fmaxf(m, fabsf(v[k]));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
  return nvfp4_store8_amax(v, m, o, row, col);
}

// MS-EDEN (Quartet II; nvfp4_sim.h's nvfp4_eden): block maxima map a little above 6, so the largest values saturate
constexpr float kEdenValMax = 6.f / (17.f / 16 * 0.93f);
// each operand's share of H64's normalization (1 / sqrt(64)), so the product carries the orthonormal rotation
constexpr float kEdenMul = 1.f / 8;

// the values of 8 e2m1 codes (as e2m1x8_rn's), exact
__device__ __forceinline__ void e2m1x8_values(uint32_t codes, float (&q)[8]) {
  uint32_t h[4];
  asm("{\n"
      " .reg .b8 b0, b1, b2, b3;\n"
      " mov.b32 {b0, b1, b2, b3}, %4;\n"
      " cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      " cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      " cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      " cvt.rn.f16x2.e2m1x2 %3, b3;\n"
      "}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(codes));
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h[k]));
    q[2 * k] = f.x, q[2 * k + 1] = f.y;
  }
}

// MS-EDEN store of one 64-group of row `row` from col (% 64) by 8 lanes (lane % 8 = k holding values 8k .. 8k + 7):
// x . diag(signs) . H64 / 8 (H64 = H8 over the values x H8 over the lanes, Sylvester order over the value index), each
// lane pair's block scale from its max / kEdenValMax, values to nearest, the scale times the group's ||a||² / <a, q>
// (in block units), rounded stochastically to 3 mantissa bits. Both lanes of a pair return the block scale.
__device__ __forceinline__ float nvfp4_eden_store8(
      float (&v)[8], const kernels::Nvfp4Out& o, int64_t row, int64_t col) {
  const int lane = static_cast<int>(threadIdx.x % 8);
  const uint32_t bits = static_cast<uint32_t>(o.eden_signs >> (8 * lane));
#pragma unroll
  for (int i = 0; i < 8; ++i)
    v[i] *= (bits >> i) & 1u ? -kEdenMul : kEdenMul;
#pragma unroll
  for (int h = 1; h < 8; h *= 2)
#pragma unroll
    for (int i = 0; i < 8; i += 2 * h)
#pragma unroll
      for (int j = i; j < i + h; ++j) {
        const float a = v[j], b = v[j + h];
        v[j] = a + b, v[j + h] = a - b;
      }
#pragma unroll
  for (int h = 1; h < 8; h *= 2)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float p = __shfl_xor_sync(0xffffffff, v[i], h);
      v[i] = lane & h ? p - v[i] : v[i] + p;
    }
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    m = fmaxf(m, fabsf(v[i]));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
  const float s = nvfp4_round_scale(m * (1.f / kEdenValMax)), to_q = s > 0.f ? nvfp4_rcp(s) : 0.f;
  float a[8], q[8], num = 0.f, den = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    a[i] = v[i] * to_q;
  const uint32_t codes = e2m1x8_rn(a);
  e2m1x8_values(codes, q);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    num = fmaf(a[i], a[i], num), den = fmaf(a[i], q[i], den);
#pragma unroll
  for (int h = 1; h < 8; h *= 2) {
    num += __shfl_xor_sync(0xffffffff, num, h);
    den += __shfl_xor_sync(0xffffffff, den, h);
  }
  const float cs = s * (den > 0.f ? num / den : 1.f);
  const int64_t i = row * o.ld + col;
  const uint32_t r = nvfp4_random(o.seed, (o.index0 + i) / 16);
  const float sr = __uint_as_float((__float_as_uint(cs) + (r & 0xfffffu)) & 0xfff00000u);
  *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(o.data) + i / 2) = codes;
  if (col % 16 == 0)
    nvfp4_store_scale(o, row, col, sr);
  return sr;
}

__device__ __forceinline__ unsigned nvfp4_slot() {
  return (blockIdx.x + blockIdx.y * gridDim.x) % kernels::kNvfp4Slots;
}

// The warp's max block scale into o.smax. All 32 lanes.
__device__ __forceinline__ void nvfp4_smax(const kernels::Nvfp4Out& o, float s) {
  const unsigned m = __reduce_max_sync(0xffffffff, __float_as_uint(s)); // non-negative floats order as unsigned
  if (threadIdx.x % 32 == 0 && m > 0)
    atomicMax(o.smax + nvfp4_slot(), static_cast<unsigned long long>(o.epoch) << 32 | m);
}

// The thread block's max block scale into o.smax, one atomic. All threads, whole warps.
__device__ __forceinline__ void nvfp4_smax_block(const kernels::Nvfp4Out& o, float s) {
  __shared__ unsigned block_max;
  if (threadIdx.x == 0)
    block_max = 0;
  __syncthreads();
  const unsigned m = __reduce_max_sync(0xffffffff, __float_as_uint(s));
  if (threadIdx.x % 32 == 0 && m > 0)
    atomicMax(&block_max, m);
  __syncthreads();
  if (threadIdx.x == 0 && block_max > 0)
    atomicMax(o.smax + nvfp4_slot(), static_cast<unsigned long long>(o.epoch) << 32 | block_max);
}

} // namespace nanochat
