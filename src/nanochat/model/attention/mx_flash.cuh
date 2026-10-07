// MXFP8 flash attention's device helpers: block-scaled mma (sm_120a) and the input quantizers (included by .cu files
// only).
#pragma once

#include <cstdint>

#include "nanochat/model/attention/flash.cuh"
#include "nanochat/model/fp8/mx_kernel.cuh"

namespace nanochat::flash {

constexpr uint32_t kPScale = 0x77u; // ue8m0 2^-8: P is stored as 256 p

// chunk c of row r, for e4m3 rows of RowBytes (32, 64 or 128): conflict-free ldmatrix and cp.async
template <int RowBytes>
__device__ __forceinline__ uint32_t swz8(int r, int c) {
  constexpr int kChunks = RowBytes / 16, kRowsPerLine = 128 / RowBytes;
  return r * RowBytes + ((c ^ ((r / kRowsPerLine) % kChunks)) << 4);
}

// Rows x RowBytes from global (ld bytes apart) into swizzled shared memory. A thread's chunk column and swizzle are
// the same on every pass (the swizzle repeats every 8 rows), so each pass only adds constant offsets.
template <int Threads, int RowBytes, int Rows>
__device__ __forceinline__ void load_rows8(uint32_t dst, const uint8_t* src, int64_t ld) {
  constexpr int kChunks = RowBytes / 16, kItems = Rows * kChunks, kStep = Threads / kChunks; // rows per pass
  static_assert(Threads % kChunks == 0 && (kItems <= Threads || (kItems % Threads == 0 && kStep % 8 == 0)));
  const int tid = static_cast<int>(threadIdx.x);
  if (kItems < Threads && tid >= kItems)
    return;
  const int r = tid / kChunks, c = tid % kChunks;
  const uint32_t d = dst + swz8<RowBytes>(r, c);
  const uint8_t* s = src + r * ld + c * 16;
#pragma unroll
  for (int i = 0; i < (kItems + Threads - 1) / Threads; ++i)
    cp_async16(d + i * kStep * RowBytes, s + i * kStep * ld);
}

// Bytes (a multiple of 16) from global into shared memory
template <int Threads, int Bytes>
__device__ __forceinline__ void load_bytes(uint32_t dst, const void* src) {
  constexpr int kItems = Bytes / 16;
  static_assert(Bytes % 16 == 0 && (kItems <= Threads || kItems % Threads == 0));
  const int tid = static_cast<int>(threadIdx.x);
  if (kItems < Threads && tid >= kItems)
    return;
#pragma unroll
  for (int i = 0; i < (kItems + Threads - 1) / Threads; ++i)
    cp_async16(dst + (tid + i * Threads) * 16, static_cast<const uint8_t*>(src) + (tid + i * Threads) * 16);
}

// Scales along tokens: a row of 128 dims per 32 tokens, dim 16 j + 8 h + g at byte g * 16 + 2 j + h, so the 16 dims
// an mma thread (g = lane / 4) needs over all n8 tiles are one 16-byte load.
__device__ __forceinline__ int t_scale_pos(int d) {
  return d % 8 * 16 + d / 16 * 2 + d / 8 % 2;
}

// the thread's 16 scale bytes of a row: n8 tile nt (dims 8 nt..) is byte nt % 4 of w[nt / 4]
__device__ __forceinline__ void load_t_scales(uint32_t (&w)[4], const uint8_t* row) {
  const uint4 v = *reinterpret_cast<const uint4*>(row + threadIdx.x % 32 / 4 * 16);
  w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
}

// d += (A 2^sfa) (B 2^sfb), e4m3 m16n8k32. sfa: row lane/4 + 8 (lane % 2)'s exponents, sfb: column lane/4's; the
// k-step's exponent is byte byte_a / byte_b.
__device__ __forceinline__ void mma_mx(
      float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb, uint16_t byte_a,
      uint16_t byte_b) {
  const uint16_t tid = 0;
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(byte_a), "h"(tid),
                 "r"(sfb), "h"(byte_b), "h"(tid));
}

// four floats to e4m3 (round to nearest, saturating), a in the low byte
__device__ __forceinline__ uint32_t pack_e4m3(float a, float b, float c, float d) {
  uint32_t out;
  asm("{\n.reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n}"
      : "=r"(out)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return out;
}

// Four m16n8 accumulator tiles c[0..3] (columns 32 kk .. + 32) as an e4m3 A fragment, times mul[0] (row g) and mul[1]
// (row g + 8). Row g's positions 4t..4t+3 hold columns {2t, 2t+1, 8+2t, 9+2t} of each 16: the B operand's k (tokens)
// must be stored in that order (store_t_tile).
__device__ __forceinline__ void pack_a8(
      uint32_t (&a)[4], const float (&c0)[4], const float (&c1)[4], const float (&c2)[4], const float (&c3)[4],
      float mul0, float mul1) {
  a[0] = pack_e4m3(c0[0] * mul0, c0[1] * mul0, c1[0] * mul0, c1[1] * mul0);
  a[1] = pack_e4m3(c0[2] * mul1, c0[3] * mul1, c1[2] * mul1, c1[3] * mul1);
  a[2] = pack_e4m3(c2[0] * mul0, c2[1] * mul0, c3[0] * mul0, c3[1] * mul0);
  a[3] = pack_e4m3(c2[2] * mul1, c2[3] * mul1, c3[2] * mul1, c3[3] * mul1);
}

// As pack_a8 with a dynamic MX scale per row (amax over the 32 columns); returns the sfa for mma_mx (byte 0).
__device__ __forceinline__ uint32_t pack_a8_scaled(
      uint32_t (&a)[4], const float (&c0)[4], const float (&c1)[4], const float (&c2)[4], const float (&c3)[4]) {
  float am0 = 0.0f, am1 = 0.0f;
  const float* cs[4] = {c0, c1, c2, c3};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    am0 = fmaxf(am0, fmaxf(fabsf(cs[i][0]), fabsf(cs[i][1])));
    am1 = fmaxf(am1, fmaxf(fabsf(cs[i][2]), fabsf(cs[i][3])));
  }
#pragma unroll
  for (int off = 1; off < 4; off *= 2) {
    am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, off));
    am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, off));
  }
  const int e0 = mx_exponent(am0), e1 = mx_exponent(am1);
  pack_a8(a, c0, c1, c2, c3, mx_multiplier(e0), mx_multiplier(e1));
  return static_cast<uint32_t>(threadIdx.x % 2 ? e1 : e0);
}

// Quantization of the attention's inputs and dout. Along head_dim ("rows"): data (B, T, heads, 128) e4m3, scale
// (B, heads, T) u32, byte j = dims [32j, 32j + 32). Along tokens ("t"): data (B, heads, 128, T), tokens permuted in
// each 16 as [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] (pack_a8's order), scale (B, heads, T / 32, 128) per 32 tokens,
// dims ordered as t_scale_pos.

constexpr int kTTokens = 64;    // tokens per tile
constexpr int kTThreads = 256;  // tile threads: (dim, 32-token half) each for store_t_tile
constexpr int kTPitch = kD + 8; // bf16 per tile row: 16-byte rows, conflict-free column reads
using TTile = __nv_bfloat16[kTTokens][kTPitch];

// 64 tokens x 128 (tokens x_ld elements apart) into the tile
__device__ __forceinline__ void load_t_tile(TTile& tile, const __nv_bfloat16* x, int64_t x_ld) {
#pragma unroll
  for (int i = 0; i < kTTokens * kD / 8 / kTThreads; ++i) {
    const int idx = static_cast<int>(threadIdx.x) + i * kTThreads, t = idx / (kD / 8), c = idx % (kD / 8);
    *reinterpret_cast<uint4*>(&tile[t][c * 8]) = *reinterpret_cast<const uint4*>(x + t * x_ld + c * 8);
  }
}

// the tile as tokens n * 64 .. of head bh (B * heads + h) along tokens
__device__ __forceinline__ void store_t_tile(
      const TTile& tile, uint8_t* data, uint8_t* scale, int64_t bh, int n, int64_t T) {
  const int d = static_cast<int>(threadIdx.x) % kD, half = static_cast<int>(threadIdx.x) / kD;
  float x[32];
  float amax = 0.0f;
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    x[t] = __bfloat162float(tile[half * 32 + t][d]);
    amax = fmaxf(amax, fabsf(x[t]));
  }
  const int e = mx_exponent(amax);
  const float mul = mx_multiplier(e);
  const auto token = [](int p) {
    const int i = p % 16;
    return p / 16 * 16 + 2 * (i / 4) + i % 2 + 8 * (i % 4 / 2);
  };
  uint32_t w[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    w[j] = pack_e4m3(
          x[token(4 * j)] * mul, x[token(4 * j + 1)] * mul, x[token(4 * j + 2)] * mul, x[token(4 * j + 3)] * mul);
  auto* dst = reinterpret_cast<uint4*>(data + (bh * kD + d) * T + static_cast<int64_t>(n) * kTTokens + half * 32);
  dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
  dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
  scale[(bh * (T / 32) + 2 * n + half) * kD + t_scale_pos(d)] = static_cast<uint8_t>(e);
}

// a row of 128 held by a warp (lane: dims 4 lane .. 4 lane + 3) along head_dim: 128 e4m3 and the scale word
__device__ __forceinline__ void store_row(const float (&x)[4], uint8_t* data, uint32_t* scale_word) {
  const int lane = static_cast<int>(threadIdx.x % 32);
  float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int off = 1; off < 8; off *= 2) // 8 lanes per MX block
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const int e = mx_exponent(amax);
  const float mul = mx_multiplier(e);
  reinterpret_cast<uint32_t*>(data)[lane] = pack_e4m3(x[0] * mul, x[1] * mul, x[2] * mul, x[3] * mul);
  uint32_t word = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    word |= static_cast<uint32_t>(__shfl_sync(0xffffffffu, e, 8 * j)) << (8 * j);
  if (lane == 0)
    *scale_word = word;
}

// row r of the tile, a warp's 4 values per lane as store_row takes them
__device__ __forceinline__ void tile_row(const TTile& tile, int r, float (&x)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(&tile[r][4 * (threadIdx.x % 32)]);
  const auto* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
  const float2 a = __bfloat1622float2(p[0]), b = __bfloat1622float2(p[1]);
  x[0] = a.x, x[1] = a.y, x[2] = b.x, x[3] = b.y;
}

// the tile as tokens t0 .. t0 + 63 of head h along head_dim (a warp per row)
__device__ __forceinline__ void store_rows_tile(
      const TTile& tile, uint8_t* data, uint32_t* scale, int b, int64_t t0, int h, int heads, int64_t T) {
  for (int r = static_cast<int>(threadIdx.x) / 32; r < kTTokens; r += kTThreads / 32) {
    float x[4];
    tile_row(tile, r, x);
    const int64_t t = t0 + r;
    store_row(x, data + ((b * T + t) * heads + h) * kD, scale + (static_cast<int64_t>(b) * heads + h) * T + t);
  }
}

// One row of 128 as a warp holds it after rotary_norm (lane: columns 2 lane + {0, 1} and 64 + 2 lane + {0, 1}),
// along head_dim as store_row.
__device__ __forceinline__ void store_row_halves(
      float a0, float a1, float b0, float b1, uint8_t* data, uint32_t* scale_word) {
  const int lane = static_cast<int>(threadIdx.x % 32);
  float amax_a = fmaxf(fabsf(a0), fabsf(a1)), amax_b = fmaxf(fabsf(b0), fabsf(b1));
#pragma unroll
  for (int off = 1; off < 16; off *= 2) { // 16 lanes per MX block
    amax_a = fmaxf(amax_a, __shfl_xor_sync(0xffffffffu, amax_a, off));
    amax_b = fmaxf(amax_b, __shfl_xor_sync(0xffffffffu, amax_b, off));
  }
  const int ea = mx_exponent(amax_a), eb = mx_exponent(amax_b);
  const float ma = mx_multiplier(ea), mb = mx_multiplier(eb);
  const uint32_t w = pack_e4m3(a0 * ma, a1 * ma, b0 * mb, b1 * mb);
  reinterpret_cast<uint16_t*>(data)[lane] = static_cast<uint16_t>(w);
  reinterpret_cast<uint16_t*>(data + kD / 2)[lane] = static_cast<uint16_t>(w >> 16);
  const uint32_t e0 = __shfl_sync(0xffffffffu, ea, 0), e1 = __shfl_sync(0xffffffffu, ea, 16);
  const uint32_t e2 = __shfl_sync(0xffffffffu, eb, 0), e3 = __shfl_sync(0xffffffffu, eb, 16);
  if (lane == 0)
    *scale_word = e0 | e1 << 8 | e2 << 16 | e3 << 24;
}

} // namespace nanochat::flash
