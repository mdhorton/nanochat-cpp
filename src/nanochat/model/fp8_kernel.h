// Fused tensorwise FP8 quantization (no torch headers). Same math as fp8.py's _to_fp8, bit for bit:
// amax = max|x|, scale = float(fp8_max * (1 / max(double(amax), 1e-12))), q = fp8(clamp(float(x) * scale)), inv = 1 /
// scale.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

enum class Fp8Format { E4M3, E5M2 };

// x: (rows, cols) contiguous bf16 (x_bf16 = true) or fp32. amax: device float scratch, or max|x| already when
// amax_ready. Writes *inv_scale (device float), and out (rows, cols) and/or out_t (cols, rows, the transpose) when
// non-null.
void quantize_fp8(
      const void* x, bool x_bf16, int64_t rows, int64_t cols, Fp8Format format, void* out, void* out_t, float* amax,
      float* inv_scale, cudaStream_t stream, bool amax_ready = false);

// quantize_fp8 of relu(h).square() as E4M3, read from bf16 h without writing the square.
void quantize_fp8_relu_square(
      const void* h, int64_t rows, int64_t cols, void* out, void* out_t, float* amax, float* inv_scale,
      cudaStream_t stream);

// NVFP4 written in place of MX data (nvfp4.h's Nvfp4Target): two e2m1 per byte, and per 16 values along a row
// its block scale, e4m3's mantissa at any exponent, as bf16; nvfp4_finish then picks the power-of-two tensor scale
// from their max (smax) and stores them as ue4m3.
inline constexpr int kNvfp4Slots = 32; // smax's, spreading the atomics

struct Nvfp4Out {
  void* data;         // (rows, ld / 2) bytes at this view; null: MX
  int64_t ld;         // row stride in values
  void* scale16;      // bf16 at this view, in the ue4m3 scales' swizzled layout (ld / 64 tiles per 128 rows)
  unsigned* smax;     // kNvfp4Slots: the whole tensor's max block scale is their max (float bits, atomicMax)
  int64_t index0;     // this view's first value's index in the whole tensor: stochastic rounding's counter
  uint64_t seed;      // stochastic rounding's
  uint32_t rht_signs; // bit i: the random Hadamard's row i negated
  bool rht, stochastic;
};

// MXFP8 (OCP MX) E4M3: one power-of-two (e8m0) scale per 32 consecutive values of a row, the smallest with
// max|block| * 2^-e <= 448, stored in cuBLAS's swizzled layout (128 rows x 4 blocks per 512-byte tile).
struct MxOut {
  void* data;          // e4m3, null: not written
  int64_t ld;          // row stride of data
  void* scale;         // e8m0, at this tensor's first tile
  int64_t scale_tiles; // tiles per 128 rows in the whole scale buffer (its blocks per row / 4)
  Nvfp4Out fp4{};      // NVFP4 instead (data null)
};

// x: (rows, cols) contiguous bf16 (x_bf16) or fp32, rows % 32 == 0, cols % 64 == 0. out: x, scaled along rows;
// out_t: x's transpose (cols, rows), scaled along its rows. relu_square: quantizes bf16(relu(x)^2), bf16 x only.
void quantize_mx(
      const void* x, bool x_bf16, int64_t rows, int64_t cols, MxOut out, MxOut out_t, bool relu_square,
      cudaStream_t stream);

// quantize_mx of relu^2's input gradient dh = bf16(h > 0 ? g * 2h : 0), read from bf16 g, h without writing dh.
void quantize_mx_relu_square_bwd(
      const void* g, const void* h, int64_t rows, int64_t cols, MxOut out, MxOut out_t, cudaStream_t stream);

} // namespace nanochat::kernels
