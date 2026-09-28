// MXFP8 flash attention's input quantization, shared by the kernels that write it (included by .cu files only).
#pragma once

#include <cstdint>

#include <cuda_bf16.h>

#include "nanochat/model/mx_flash_kernel.h"
#include "nanochat/model/mx_kernel.cuh"

namespace nanochat::mx_flash {

constexpr int kD = kernels::kMxFlashHeadDim;
constexpr int kVtTokens = 64;    // tokens per Vᵀ tile (the attention's key tile)
constexpr int kVtThreads = 256;  // (dim, 32-token half) per thread
constexpr int kVtPitch = kD + 8; // bf16 per tile row: 16-byte rows, conflict-free column reads
using VtTile = __nv_bfloat16[kVtTokens][kVtPitch];

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

// Vᵀ tile n of head (b, h) from 64 tokens x 128 dims in shared memory (kVtThreads threads): MX blocks of 32 tokens
// per dim, position p of each 16 holding token [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15][p]. Layouts as
// kernels::mx_flash_quantize_vt.
__device__ __forceinline__ void store_vt_tile(
      const VtTile& tile, uint8_t* vt, uint8_t* scale, int64_t bh, int n, int64_t T) {
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
  auto* dst = reinterpret_cast<uint4*>(vt + (bh * kD + d) * T + static_cast<int64_t>(n) * kVtTokens + half * 32);
  dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
  dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
  scale[((bh * (T / kVtTokens) + n) * kD + d) * 2 + half] = static_cast<uint8_t>(e);
}

// One row of 128 as a warp holds it after rotary_norm (lane: columns 2 lane + {0, 1} and 64 + 2 lane + {0, 1}), to
// data (128 e4m3) and its scale word (byte j: dims [32j, 32j + 32)), as kernels::mx_flash_quantize_rows.
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

} // namespace nanochat::mx_flash
