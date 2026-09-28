// MXFP8 device helpers for the flash-attention backward (included by .cu files only; block-scaled mma needs sm_120a).
#pragma once

#include <cstdint>

#include "nanochat/model/flash.cuh"
#include "nanochat/model/mx_kernel.cuh"

namespace nanochat::flash {

constexpr uint32_t kPScale = 0x77u; // ue8m0 2^-8: P is stored as 256 p

// chunk c of row r, for e4m3 rows of RowBytes (32, 64 or 128): conflict-free ldmatrix and cp.async
template <int RowBytes>
__device__ __forceinline__ uint32_t swz8(int r, int c) {
  constexpr int kChunks = RowBytes / 16, kRowsPerLine = 128 / RowBytes;
  return r * RowBytes + ((c ^ ((r / kRowsPerLine) % kChunks)) << 4);
}

// rows x RowBytes from global (ld bytes apart) into swizzled shared memory
template <int Threads, int RowBytes>
__device__ __forceinline__ void load_rows8(uint32_t dst, const uint8_t* src, int64_t ld, int rows) {
  constexpr int kChunks = RowBytes / 16;
  for (int idx = static_cast<int>(threadIdx.x); idx < rows * kChunks; idx += Threads) {
    const int r = idx / kChunks, c = idx % kChunks;
    cp_async16(dst + swz8<RowBytes>(r, c), src + r * ld + c * 16);
  }
}

// bytes (a multiple of 16) from global into shared memory
template <int Threads>
__device__ __forceinline__ void load_bytes(uint32_t dst, const void* src, int bytes) {
  for (int idx = static_cast<int>(threadIdx.x); idx < bytes / 16; idx += Threads)
    cp_async16(dst + idx * 16, static_cast<const uint8_t*>(src) + idx * 16);
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

} // namespace nanochat::flash
